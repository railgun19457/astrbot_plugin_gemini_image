"""
Gemini Image Generation Plugin
使用 Gemini 系列模型进行图像生成的插件
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Coroutine
from typing import Any

import aiohttp
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config.astrbot_config import AstrBotConfig

from .gemini_generator import GeminiImageGenerator


@pydantic_dataclass
class GeminiImageGenerationTool(FunctionTool[AstrAgentContext]):
    """统一的图像生成工具，支持文生图和图生图"""

    name: str = "gemini_generate_image"
    description: str = "使用 Gemini 模型生成图片"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生成图片时使用的详细提示词(推荐英文或中文)",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比",
                    "enum": [
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                },
                "resolution": {
                    "type": "string",
                    "description": "图片分辨率，仅 gemini-3-pro-image-preview 模型支持",
                    "enum": ["1K", "2K", "4K"],
                },
                "use_reference_image": {
                    "type": "boolean",
                    "description": "是否使用参考图片,默认: false",
                },
                "reference_image_index": {
                    "type": "number",
                    "description": "参考图片的索引,从0开始。仅在 use_reference_image=true 时有效。默认使用最新的图片(0)",
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    def __post_init__(self):
        """动态更新 description 以包含当前模型信息"""
        if self.plugin and hasattr(self.plugin, "model"):
            model = self.plugin.model
            self.description = f"使用 Gemini 模型生成图片。当前模型: {model}"

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        prompt = kwargs.get("prompt", "")
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        resolution = kwargs.get("resolution", "1K")
        use_reference_image = kwargs.get("use_reference_image", False)
        image_index = int(kwargs.get("reference_image_index", 0))

        if not prompt:
            return "请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            try:
                plugin = context.context.context
            except AttributeError:
                plugin = None

        if not plugin:
            return "❌ 插件初始化失败，请联系管理员"

        # 获取事件
        event = None
        try:
            event = context.context.event
        except AttributeError:
            pass

        if not event:
            return "❌ 无法获取当前消息上下文"

        # 根据参数决定是否使用参考图片
        image_data = None
        mime_type = None

        if use_reference_image:
            recent_images = plugin.get_recent_images(event.unified_msg_origin)
            if not recent_images or image_index >= len(recent_images):
                available_count = len(recent_images) if recent_images else 0
                return f"❌ 未找到参考图片！\n\n📷 当前可用图片数: {available_count}\n💡 请先发送图片，然后使用图生图功能"

            ref_image = recent_images[image_index]
            # 从 URL 下载图片
            result = await plugin._download_image(ref_image["url"])
            if not result:
                return "❌ 下载参考图片失败，请重试"
            image_data, mime_type = result

        # 创建异步任务,在后台生成图片
        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                image_data=image_data,
                mime_type=mime_type,
                unified_msg_origin=event.unified_msg_origin,
                use_reference_image=use_reference_image,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
        )

        return "[图片生成任务已启动，请等待结果]"


class GeminiImagePlugin(Star):
    """Gemini 图像生成插件"""

    # 配置验证常量
    DEFAULT_CACHE_TTL = 3600  # 默认缓存时间 (秒)
    MAX_CACHE_TTL = 86400  # 最大缓存时间 (24小时)
    DEFAULT_MAX_CACHE_COUNT = 100  # 默认最大缓存数量
    MAX_CACHE_COUNT = 1000  # 最大缓存数量
    DEFAULT_MAX_IMAGE_SIZE_MB = 10  # 默认最大图片大小 (MB)
    MAX_IMAGE_SIZE_MB = 50  # 最大图片大小 (MB)
    DEFAULT_MAX_CONCURRENT_GENERATIONS = 3  # 默认最大并发生成数
    MAX_CONCURRENT_GENERATIONS = 10  # 最大并发生成数
    DEFAULT_MAX_IMAGES_PER_SESSION = 5  # 默认每会话最大图片数
    IMAGE_CACHE_TTL = 3600  # 图片缓存过期时间 (秒)

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()

        # 读取配置
        self._load_config()

        # 初始化生成器
        self.generator = GeminiImageGenerator(
            api_keys=self.api_keys,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            cache_ttl=self.cache_ttl,
            max_cache_count=self.max_cache_count,
        )

        # 存储最近收到的图片 {session_id: [{"data": bytes, "mime_type": str, "timestamp": float}]}
        self.recent_images: dict[str, list[dict]] = {}
        self.max_images_per_session = self.DEFAULT_MAX_IMAGES_PER_SESSION
        # max_image_size 已在 _validate_config 中设置
        self.image_cache_ttl = self.IMAGE_CACHE_TTL  # 图片缓存过期时间（秒）

        # 异步任务追踪
        self.background_tasks: set[asyncio.Task] = set()

        # 并发控制 - 使用验证后的值
        self._generation_semaphore = asyncio.Semaphore(self.max_concurrent_generations)

        # 注册工具到 LLM
        if self.enable_llm_tool:
            # 将插件实例注入到工具中，方便工具在执行时访问生成器和缓存
            self.context.add_llm_tools(GeminiImageGenerationTool(plugin=self))
            logger.info("[Gemini Image] 已注册统一的图像生成工具")

        logger.info(f"[Gemini Image] 插件已加载，使用模型: {self.model}")

    def _load_config(self):
        """加载配置"""
        use_system_provider = self.config.get("use_system_provider", True)
        provider_id = (self.config.get("provider_id", "") or "").strip()

        # 尝试从系统提供商加载配置
        if use_system_provider and provider_id:
            if not self._load_provider_config(provider_id):
                self._load_default_config()
        else:
            if use_system_provider:
                logger.warning("[Gemini Image] 未配置提供商 ID，将使用插件配置")
            self._load_default_config()

        # 加载模型配置
        self.model = self._load_model_config()

        # 加载其他配置
        self.timeout = self.config.get("timeout", 120)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.max_cache_count = self.config.get("max_cache_count", 100)
        self.enable_llm_tool = self.config.get("enable_llm_tool", True)
        self.default_aspect_ratio = self.config.get("default_aspect_ratio", "1:1")
        self.default_resolution = self.config.get("default_resolution", "1K")

        self._validate_config()

    def _load_provider_config(self, provider_id: str) -> bool:
        """从系统提供商加载配置，返回是否成功"""
        provider = self.context.get_provider_by_id(provider_id)
        if not provider:
            logger.warning(f"[Gemini Image] 未找到提供商 {provider_id}，将使用插件配置")
            return False

        api_keys, api_base = self._extract_provider_credentials(provider)
        if not api_keys:
            logger.warning(
                f"[Gemini Image] 提供商 {provider_id} 未提供可用的 API Key，将使用插件配置"
            )
            return False

        self.api_keys = api_keys
        self.base_url = api_base or "https://generativelanguage.googleapis.com"
        logger.info(
            f"[Gemini Image] 使用系统提供商: {provider_id}，API Keys 数量: {len(self.api_keys)}"
        )
        return True

    def _load_model_config(self) -> str:
        """加载模型配置"""
        model = self.config.get("model", "gemini-2.0-flash-exp-image-generation")
        if model != "custom":
            return model

        custom_model = self.config.get("custom_model", "").strip()
        if custom_model:
            logger.info(f"[Gemini Image] 使用自定义模型: {custom_model}")
            return custom_model

        logger.warning("[Gemini Image] 选择了 custom 但未配置 custom_model，将使用默认模型")
        return "gemini-2.0-flash-exp-image-generation"

    def _validate_numeric_config(
        self,
        value: Any,
        name: str,
        min_val: float | int,
        max_val: float | int,
        default: float | int,
    ) -> float | int:
        """通用数值配置验证函数"""
        if not isinstance(value, (int, float)) or value <= min_val:
            logger.warning(
                f"[Gemini Image] 无效的{name}: {value}，使用默认值 {default}"
            )
            return default
        elif value > max_val:
            logger.warning(f"[Gemini Image] {name}过大: {value}，限制为 {max_val}")
            return max_val
        return value

    def _validate_config(self) -> None:
        """验证配置值的合理性"""
        self.timeout = self._validate_numeric_config(
            self.timeout, "超时时间", 0, 600, 120
        )
        self.cache_ttl = self._validate_numeric_config(
            self.cache_ttl, "缓存时间", 0, self.MAX_CACHE_TTL, self.DEFAULT_CACHE_TTL
        )
        self.max_cache_count = self._validate_numeric_config(
            self.max_cache_count,
            "最大缓存数量",
            0,
            self.MAX_CACHE_COUNT,
            self.DEFAULT_MAX_CACHE_COUNT,
        )

        # 验证最大图片大小
        max_image_size_mb = self.config.get(
            "max_image_size_mb", self.DEFAULT_MAX_IMAGE_SIZE_MB
        )
        max_image_size_mb = self._validate_numeric_config(
            max_image_size_mb,
            "最大图片大小",
            0,
            self.MAX_IMAGE_SIZE_MB,
            self.DEFAULT_MAX_IMAGE_SIZE_MB,
        )
        self.max_image_size = max_image_size_mb * 1024 * 1024

        # 验证并发生成数
        max_concurrent = self.config.get(
            "max_concurrent_generations", self.DEFAULT_MAX_CONCURRENT_GENERATIONS
        )
        self.max_concurrent_generations = self._validate_numeric_config(
            max_concurrent,
            "并发生成数",
            0,
            self.MAX_CONCURRENT_GENERATIONS,
            self.DEFAULT_MAX_CONCURRENT_GENERATIONS,
        )

        # 验证每分钟请求数
        max_requests_per_minute = self.config.get("max_requests_per_minute", 5)
        self._validate_numeric_config(max_requests_per_minute, "每分钟请求数", 0, 60, 5)
        # 注意：这里只是验证，实际的请求限制需要额外的实现

        # 验证默认宽高比
        valid_aspect_ratios = [
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ]
        if self.default_aspect_ratio not in valid_aspect_ratios:
            logger.warning(
                f"[Gemini Image] 无效的默认宽高比: {self.default_aspect_ratio}，使用默认值 1:1"
            )
            self.default_aspect_ratio = "1:1"

        # 验证默认分辨率
        valid_resolutions = ["1K", "2K", "4K"]
        if self.default_resolution not in valid_resolutions:
            logger.warning(
                f"[Gemini Image] 无效的默认分辨率: {self.default_resolution}，使用默认值 1K"
            )
            self.default_resolution = "1K"

    def _load_default_config(self):
        """加载默认配置"""
        api_key = self.config.get("api_key", "")
        # 支持单个key或多个key
        if isinstance(api_key, list):
            self.api_keys = [k for k in api_key if k]
        elif isinstance(api_key, str) and api_key:
            self.api_keys = [api_key]
        else:
            self.api_keys = []

        self.base_url = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com"
        )
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

    def _extract_provider_credentials(
        self, provider: object
    ) -> tuple[list[str], str | None]:
        """从 Provider 实例提取 API Keys 与 Base URL"""
        provider_config = getattr(provider, "provider_config", {}) or {}

        # 提取 API Keys
        api_keys = self._extract_api_keys(provider_config)

        # 提取 API Base URL
        api_base = (
            getattr(provider, "api_base", None)
            or provider_config.get("api_base")
            or provider_config.get("api_base_url")
        )
        if isinstance(api_base, str):
            api_base = api_base.rstrip("/")

        return api_keys, api_base

    def _extract_api_keys(self, provider_config: dict) -> list[str]:
        """从提供商配置中提取 API Keys"""
        # 尝试多种可能的 key 字段
        for key_field in ["key", "keys", "api_key", "access_token"]:
            keys = provider_config.get(key_field)
            if not keys:
                continue

            if isinstance(keys, str) and keys:
                return [keys]
            elif isinstance(keys, list):
                return [k for k in keys if k]

        return []

    @filter.command("img")
    async def generate_image_command(self, event: AstrMessageEvent):
        """生成图片指令

        用法:
        /img <提示词> - 文生图
        /img <提示词> (引用包含图片的消息) - 图生图
        """
        prompt = event.message_str.strip()
        if not prompt:
            yield event.plain_result(
                "❌ 请提供图片生成的提示词！\n\n📝 用法示例:\n• /img 一只可爱的小猫\n• /img 未来城市的风景"
            )
            return

        # 获取图片数据
        image_data, mime_type = await self._get_reference_image(event)
        mode = "图生图" if image_data else "文生图"
        yield event.plain_result(f"已开始{mode}任务")

        # 创建异步任务,在后台生成图片
        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                image_data=image_data,
                mime_type=mime_type,
                unified_msg_origin=event.unified_msg_origin,
                use_reference_image=image_data is not None,
                aspect_ratio=self.default_aspect_ratio,
                resolution=self.default_resolution,
            )
        )

    async def _get_reference_image(
        self, event: AstrMessageEvent
    ) -> tuple[bytes | None, str | None]:
        """获取参考图片（优先从消息中获取，失败则从缓存获取）"""
        # 从消息链中查找图片
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                result = await self._download_image_from_component(component)
                if result:
                    return result

        # 如果消息中没有图片或下载失败，从缓存 URL 下载
        recent_images = self.get_recent_images(event.unified_msg_origin)
        if recent_images:
            first_image = recent_images[0]
            return await self._download_image(first_image["url"])

        return None, None

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听消息，缓存用户发送的图片 URL"""
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                image_url = component.url or component.file
                if image_url:
                    self._remember_user_image_url(
                        event.unified_msg_origin, image_url, "image/jpeg"
                    )

    def get_recent_images(self, session_id: str) -> list[dict]:
        """获取会话的最近图片"""
        # 先清理过期图片
        self._cleanup_expired_images(session_id)
        return self.recent_images.get(session_id, [])

    def _cleanup_expired_images(self, session_id: str | None = None) -> None:
        """清理过期图片"""
        current_time = time.time()
        sessions = [session_id] if session_id else list(self.recent_images.keys())

        for sid in sessions:
            if sid not in self.recent_images:
                continue

            # 过滤未过期的图片
            valid_images = [
                img
                for img in self.recent_images[sid]
                if current_time - img["timestamp"] < self.image_cache_ttl
            ]

            # 更新或删除会话
            if valid_images:
                if len(valid_images) != len(self.recent_images[sid]):
                    logger.debug(
                        f"[Gemini Image] 清理会话 {sid} 的 {len(self.recent_images[sid]) - len(valid_images)} 张过期图片"
                    )
                    self.recent_images[sid] = valid_images
            else:
                del self.recent_images[sid]

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """统一创建后台任务并追踪生命周期"""

        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    async def _download_image_from_component(
        self, component: Comp.Image
    ) -> tuple[bytes, str] | None:
        """从消息组件下载图片"""
        image_url = component.url or component.file
        return await self._download_image(image_url)

    async def _download_image(self, image_url: str | None) -> tuple[bytes, str] | None:
        """下载图片并返回数据与 MIME 类型"""
        if not image_url:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        logger.error(f"[Gemini Image] 下载图片失败: {resp.status} - {image_url}")
                        return None

                    image_data = await resp.read()

                    # 验证图片大小
                    if len(image_data) > self.max_image_size:
                        logger.warning(
                            f"[Gemini Image] 图片大小超过限制: {len(image_data)} > {self.max_image_size} bytes"
                        )
                        return None

                    mime_type = resp.headers.get("Content-Type", "image/jpeg")
                    logger.info(f"[Gemini Image] 下载图片成功: {len(image_data)} bytes")
                    return image_data, mime_type

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(f"[Gemini Image] 下载图片时出错: {exc}")
            return None

    def _remember_user_image_url(
        self, session_id: str, image_url: str, mime_type: str | None
    ) -> None:
        """缓存用户发送的图片 URL（而非完整数据，节省内存）"""
        session_images = self.recent_images.setdefault(session_id, [])
        session_images.insert(
            0,
            {
                "url": image_url,
                "mime_type": mime_type or "image/jpeg",
                "timestamp": time.time(),
            },
        )

        # 限制缓存数量
        if len(session_images) > self.max_images_per_session:
            del session_images[self.max_images_per_session :]

        logger.info(
            f"[Gemini Image] 已缓存用户图片 URL，会话 {session_id} 当前有 {len(session_images)} 张图片"
        )

        # 定期清理所有会话的过期图片（每10次缓存操作清理一次）
        if not hasattr(self, "_cache_counter"):
            self._cache_counter = 0
        self._cache_counter += 1
        if self._cache_counter >= 10:
            self._cache_counter = 0
            self._cleanup_expired_images()

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        image_data: bytes | None = None,
        mime_type: str | None = None,
        use_reference_image: bool = False,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
    ):
        """异步生成图片并发送给用户"""
        async with self._generation_semaphore:
            try:
                logger.info(
                    f"[Gemini Image] 开始异步生成任务，会话: {unified_msg_origin}，提示词: {prompt[:50]}..."
                )

                # 调用生成接口
                result_data, error = await self.generator.generate_image(
                    prompt=prompt,
                    image_data=image_data,
                    mime_type=mime_type,
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                )

                if error:
                    await self._send_error_message(unified_msg_origin, error)
                    return

                # 缓存并发送图片
                image_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()
                file_path = await self.generator.cache_image(image_id, result_data)
                await self.context.send_message(
                    unified_msg_origin, MessageChain().file_image(str(file_path))
                )

                mode = "图生图" if use_reference_image else "文生图"
                logger.info(f"[Gemini Image] {mode}任务完成，已发送给用户")

            except Exception as e:
                logger.error(f"[Gemini Image] 异步生成任务失败: {e}", exc_info=True)
                await self._send_error_message(
                    unified_msg_origin, "图片生成过程中发生未知错误，请稍后重试或联系管理员"
                )

    async def _send_error_message(self, unified_msg_origin: str, error: str):
        """发送错误消息"""
        error_msg = f"❌ 图片生成失败: {error}\n\n💡 可能的原因:\n• 提示词描述过于复杂\n• API 服务暂时不可用\n• 请稍后重试"
        logger.error(f"[Gemini Image] {error_msg}")
        try:
            await self.context.send_message(
                unified_msg_origin, MessageChain().message(error_msg)
            )
        except Exception:
            pass

    async def terminate(self):
        """插件卸载时清理资源"""
        try:
            # 取消所有后台任务
            if hasattr(self, "background_tasks"):
                pending_count = len(self.background_tasks)
                if pending_count > 0:
                    logger.info(
                        f"[Gemini Image] 正在取消 {pending_count} 个后台生成任务..."
                    )
                    for task in self.background_tasks:
                        if not task.done():
                            task.cancel()
                    # 等待所有任务取消
                    await asyncio.gather(*self.background_tasks, return_exceptions=True)
                    logger.info("[Gemini Image] 所有后台任务已取消")

            # 清理图片缓存内存
            if hasattr(self, "recent_images"):
                total_images = sum(
                    len(images) for images in self.recent_images.values()
                )
                self.recent_images.clear()
                logger.info(
                    f"[Gemini Image] 已清理内存中的图片缓存 ({total_images} 张)"
                )

            # 清理生成器资源
            if hasattr(self, "generator") and self.generator:
                # 清理生成器的图片缓存
                if hasattr(self.generator, "image_cache"):
                    cache_count = len(self.generator.image_cache)
                    self.generator.image_cache.clear()
                    logger.info(f"[Gemini Image] 已清理生成器缓存 ({cache_count} 个)")

            logger.info("[Gemini Image] 插件已卸载")
        except Exception as e:
            logger.error(f"[Gemini Image] 清理资源时出错: {e}")
