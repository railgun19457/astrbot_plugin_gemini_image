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
                    "description": "生图时使用的提示词(直接将用户发送的内容原样传递给模型)",
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
                "num_cached_images": {
                    "type": "number",
                    "description": "使用最近缓存的图片数量（当用户没有直接提供图片时）。0=不使用缓存，1=使用最新1张，2=使用最新2张，最多3张。默认: 0",
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    def __post_init__(self):
        """动态更新 description 以包含当前模型信息"""
        if self.plugin and hasattr(self.plugin, "model"):
            self.description = f"使用 Gemini 模型生成图片。当前模型: {self.plugin.model}"

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        if not (prompt := kwargs.get("prompt", "")):
            return "请提供图片生成的提示词"

        # 优先使用注入的插件实例，否则从 context 中获取
        plugin = self.plugin
        if not plugin and hasattr(context, "context") and isinstance(context.context, AstrAgentContext):
            plugin = context.context.context

        if not plugin:
            return "❌ 插件初始化失败，请联系管理员"

        # 从 AstrAgentContext 中获取 event
        event = None
        if hasattr(context, "context") and isinstance(context.context, AstrAgentContext):
            event = context.context.event

        if not event:
            return "❌ 无法获取当前消息上下文"

        # 快速验证配置
        if not plugin.generator.api_keys:
            return "❌ 未配置 API Key，无法生成图片"

        # 获取参考图片（优先从消息中获取，可选使用缓存）
        num_cached = int(kwargs.get("num_cached_images", 0))
        images_data = await plugin._get_reference_images_for_tool(
            event,
            num_cached_images=max(0, min(num_cached, 3))  # 限制在 0-3 之间
        )

        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=kwargs.get("aspect_ratio", "1:1"),
                resolution=kwargs.get("resolution", "1K"),
            )
        )

        # 返回简短确认，让 LLM 基于此生成自然的回复
        mode = "图生图" if images_data else "文生图"
        return f"已启动{mode}任务"


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
    MAX_IMAGES_PER_SESSION = 3  # 每会话最大图片数（硬编码，仅作为备用）
    IMAGE_CACHE_TTL = 3600  # 图片缓存过期时间 (秒)

    # 可用模型列表
    AVAILABLE_MODELS = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview",
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()

        # 获取系统配置中的唤醒前缀
        system_config = self.context.get_config()
        self.wake_prefixes = system_config.get("wake_prefix", ["/"])
        if not isinstance(self.wake_prefixes, list):
            self.wake_prefixes = [self.wake_prefixes]

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
            max_retry_attempts=self.max_retry_attempts,
        )

        # 存储最近收到的图片 {session_id: [{"url": str, "mime_type": str, "timestamp": float}]}
        self.recent_images: dict[str, list[dict]] = {}
        self.max_images_per_session = self.MAX_IMAGES_PER_SESSION  # 硬编码为3
        # max_image_size 已在 _validate_config 中设置
        self.image_cache_ttl = self.IMAGE_CACHE_TTL  # 图片缓存过期时间（秒）

        # 异步任务追踪
        self.background_tasks: set[asyncio.Task] = set()

        # 并发控制 - 使用验证后的值
        self._generation_semaphore = asyncio.Semaphore(self.max_concurrent_generations)

        # 启动定时清理任务
        self._cleanup_task = self.create_background_task(self._periodic_cleanup_images())

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

        if not (use_system_provider and provider_id and self._load_provider_config(provider_id)):
            if use_system_provider and not provider_id:
                logger.warning("[Gemini Image] 未配置提供商 ID，将使用插件配置")
            self._load_default_config()

        self.model = self._load_model_config()
        self.timeout = self.config.get("timeout", 300)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.max_cache_count = self.config.get("max_cache_count", 50)
        self.enable_llm_tool = self.config.get("enable_llm_tool", True)
        self.default_aspect_ratio = self.config.get("default_aspect_ratio", "1:1")
        self.default_resolution = self.config.get("default_resolution", "1K")
        self.max_retry_attempts = self.config.get("max_retry_attempts", 3)
        self.presets = self._load_presets()
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
        # 处理 base_url，移除可能的 /v1 或 /v1beta 后缀
        self.base_url = self._normalize_base_url(
            api_base or "https://generativelanguage.googleapis.com"
        )
        logger.info(
            f"[Gemini Image] 使用系统提供商: {provider_id}，API Keys 数量: {len(self.api_keys)}"
        )
        return True

    def _normalize_base_url(self, url: str) -> str:
        """规范化 base_url，移除 /v1* 后缀"""
        url = url.rstrip("/")
        # 移除所有 /v1 开头的路径段（如 /v1, /v1beta, /v1alpha 等）
        parts = url.rsplit("/", 1)
        if len(parts) == 2 and parts[1].startswith("v1"):
            return parts[0]
        return url

    def _load_model_config(self) -> str:
        """加载模型配置"""
        model = self.config.get("model", "gemini-2.0-flash-exp-image-generation")
        if model != "自定义模型":
            return model
        if custom_model := self.config.get("custom_model", "").strip():
            logger.info(f"[Gemini Image] 使用自定义模型: {custom_model}")
            return custom_model
        logger.warning("[Gemini Image] 选择了自定义模型但未配置 custom_model，将使用默认模型")
        return "gemini-2.0-flash-exp-image-generation"

    def _load_presets(self) -> dict[str, str]:
        """加载预设提示词配置

        格式: "名称:提示词"，第一个冒号前为名称，后面全部为提示词

        Returns:
            预设名称到提示词的映射字典
        """
        presets_config = self.config.get("presets", [])
        presets_dict = {}

        if not isinstance(presets_config, list):
            logger.warning("[Gemini Image] 预设配置格式错误，应为列表")
            return presets_dict

        for preset_str in presets_config:
            if not isinstance(preset_str, str):
                continue

            # 使用第一个冒号分割，前面是名称，后面全部是提示词
            if ":" not in preset_str:
                logger.warning(f"[Gemini Image] 预设格式错误（缺少冒号）: {preset_str}")
                continue

            # 只分割第一个冒号
            name, prompt = preset_str.split(":", 1)
            name = name.strip()
            prompt = prompt.strip()

            if name and prompt:
                presets_dict[name] = prompt
                logger.debug(f"[Gemini Image] 加载预设: {name}")
            else:
                logger.warning(f"[Gemini Image] 预设格式错误（名称或提示词为空）: {preset_str}")

        if presets_dict:
            logger.info(f"[Gemini Image] 已加载 {len(presets_dict)} 个预设提示词")

        return presets_dict

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
        self.timeout = self._validate_numeric_config(self.timeout, "超时时间", 0, 600, 120)
        self.cache_ttl = self._validate_numeric_config(
            self.cache_ttl, "缓存时间", 0, self.MAX_CACHE_TTL, self.DEFAULT_CACHE_TTL
        )
        self.max_cache_count = self._validate_numeric_config(
            self.max_cache_count, "最大缓存数量", 0, self.MAX_CACHE_COUNT, self.DEFAULT_MAX_CACHE_COUNT
        )

        # 验证最大图片大小
        max_image_size_mb = self._validate_numeric_config(
            self.config.get("max_image_size_mb", self.DEFAULT_MAX_IMAGE_SIZE_MB),
            "最大图片大小", 0, self.MAX_IMAGE_SIZE_MB, self.DEFAULT_MAX_IMAGE_SIZE_MB
        )
        self.max_image_size = int(max_image_size_mb * 1024 * 1024)

        # 验证并发生成数
        self.max_concurrent_generations = self._validate_numeric_config(
            self.config.get("max_concurrent_generations", self.DEFAULT_MAX_CONCURRENT_GENERATIONS),
            "并发生成数", 0, self.MAX_CONCURRENT_GENERATIONS, self.DEFAULT_MAX_CONCURRENT_GENERATIONS
        )

        # 验证重试次数
        self.max_retry_attempts = int(self._validate_numeric_config(
            self.max_retry_attempts,
            "重试次数", 0, 10, 3
        ))

        # 验证默认宽高比和分辨率
        if self.default_aspect_ratio not in ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]:
            logger.warning(f"[Gemini Image] 无效的默认宽高比: {self.default_aspect_ratio}，使用默认值 1:1")
            self.default_aspect_ratio = "1:1"

        if self.default_resolution not in ["1K", "2K", "4K"]:
            logger.warning(
                f"[Gemini Image] 无效的默认分辨率: {self.default_resolution}，使用默认值 1K"
            )
            self.default_resolution = "1K"

    def _load_default_config(self):
        """加载默认配置"""
        api_key = self.config.get("api_key", "")
        self.api_keys = (
            [k for k in api_key if k] if isinstance(api_key, list)
            else [api_key] if api_key else []
        )
        self.base_url = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com"
        ).rstrip("/")

    def _extract_provider_credentials(
        self, provider: object
    ) -> tuple[list[str], str | None]:
        """从 Provider 实例提取 API Keys 与 Base URL"""
        provider_config = getattr(provider, "provider_config", {}) or {}
        api_keys = self._extract_api_keys(provider_config)
        api_base = (
            getattr(provider, "api_base", None)
            or provider_config.get("api_base")
            or provider_config.get("api_base_url")
        )
        return api_keys, api_base.rstrip("/") if isinstance(api_base, str) else api_base

    def _extract_api_keys(self, provider_config: dict) -> list[str]:
        """从提供商配置中提取 API Keys"""
        for key_field in ["key", "keys", "api_key", "access_token"]:
            keys = provider_config.get(key_field)
            if keys:
                return [keys] if isinstance(keys, str) else [k for k in keys if k]
        return []

    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """生成图片指令

        用法:
        /生图 <提示词或预设名称> - 文生图
        /生图 <提示词或预设名称> (引用包含图片的消息) - 图生图（支持多张图片）
        /生图 <提示词或预设名称> @用户 - 使用被@用户的头像作为参考图
        """
        # 从消息链中提取纯文本（排除 At 组件）、被@的用户和被引用用户ID
        text_parts = []
        at_users = []
        replied_user_id = None

        for seg in event.get_messages():
            if isinstance(seg, Comp.Plain):
                text_parts.append(seg.text)
            elif isinstance(seg, Comp.At):
                at_users.append(str(seg.qq))
            elif isinstance(seg, Comp.Reply):
                # 获取被引用用户的ID（尝试多个可能的属性）
                replied_user_id = (
                    getattr(seg, "user_id", None) or
                    getattr(seg, "sender_id", None) or
                    getattr(seg, "qq", None)
                )
                if replied_user_id:
                    replied_user_id = str(replied_user_id)
                    logger.debug(f"[Gemini Image] 检测到引用消息，被引用用户ID: {replied_user_id}")

        # 合并纯文本
        user_input = "".join(text_parts).strip()

        # 移除指令前缀（@filter.command 不会自动去除）
        # 构建所有可能的前缀组合：wake_prefix + "生图"
        possible_prefixes = []
        for wake_prefix in self.wake_prefixes:
            # 带空格和不带空格的版本
            possible_prefixes.append(f"{wake_prefix}生图 ")
            possible_prefixes.append(f"{wake_prefix}生图")
        # 添加不带唤醒前缀的版本（某些情况下可能直接是 "生图"）
        possible_prefixes.extend(["生图 ", "生图"])

        # 按长度降序排序，优先匹配更长的前缀
        possible_prefixes.sort(key=len, reverse=True)

        for prefix in possible_prefixes:
            if user_input.startswith(prefix):
                user_input = user_input[len(prefix):].strip()
                break

        if not user_input:
            # 构建帮助信息
            help_text = "❌ 请提供图片生成的提示词或预设名称！\n\n📝 用法示例:\n• /生图 一只可爱的小猫\n• /生图 未来城市的风景"

            # 如果有预设，显示可用预设列表
            if self.presets:
                preset_names = "、".join(self.presets.keys())
                help_text += f"\n\n✨ 可用预设: {preset_names}"

            yield event.plain_result(help_text)
            return

        # 检查是否使用预设
        if user_input in self.presets:
            prompt = self.presets[user_input]
            logger.info(f"[Gemini Image] 会话 {event.unified_msg_origin} 使用预设 '{user_input}'")
            logger.debug(f"[Gemini Image] 预设内容: {prompt}")
        else:
            # 不是预设，直接使用用户输入作为提示词
            prompt = user_input

        # 获取参考图片列表（指令生图不使用缓存，只从当前消息获取）
        images_data = await self._get_reference_images_for_tool(event, num_cached_images=0)

        # 下载所有被@用户的头像作为参考图（排除被引用用户）
        if at_users:
            # 过滤掉被引用用户的ID
            filtered_at_users = [uid for uid in at_users if uid != replied_user_id]

            if filtered_at_users:
                logger.info(f"[Gemini Image] 检测到 {len(filtered_at_users)} 个@用户，正在下载头像作为参考图")
                for target_id in filtered_at_users:
                    avatar_data = await self.get_avatar(target_id)
                    if avatar_data:
                        images_data.append((avatar_data, "image/jpeg"))
                        logger.info(f"[Gemini Image] 成功添加用户 {target_id} 的头像作为参考图")
                    else:
                        logger.warning(f"[Gemini Image] 下载用户 {target_id} 的头像失败")

            if replied_user_id and replied_user_id in at_users:
                logger.debug(f"[Gemini Image] 跳过被引用用户 {replied_user_id} 的头像下载")

        mode = f"图生图({len(images_data)}张参考图)" if images_data else "文生图"

        # 如果使用了预设，在提示中显示预设名称
        if user_input in self.presets:
            yield event.plain_result(f"已开始{mode}任务（预设: {user_input}）")
        else:
            yield event.plain_result(f"已开始{mode}任务")

        # 创建异步任务,在后台生成图片
        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=self.default_aspect_ratio,
                resolution=self.default_resolution,
            )
        )

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """生图模型管理指令

        用法:
        /生图模型 - 显示可用模型列表和当前使用的模型
        /生图模型 <序号> - 切换到指定序号的模型
        """
        # 如果没有参数，显示模型列表
        if not model_index:
            model_list = "📋 可用模型列表:\n\n"
            for idx, model in enumerate(self.AVAILABLE_MODELS, 1):
                current_marker = " ✓" if model == self.model else ""
                model_list += f"{idx}. {model}{current_marker}\n"

            model_list += f"\n当前使用: {self.model}"
            model_list += "\n\n💡 使用 /生图模型 <序号> 切换模型"

            yield event.plain_result(model_list)
            return

        # 如果有参数，尝试切换模型
        try:
            index = int(model_index) - 1
            if 0 <= index < len(self.AVAILABLE_MODELS):
                new_model = self.AVAILABLE_MODELS[index]
                old_model = self.model

                # 更新模型
                self.model = new_model
                self.generator.model = new_model

                # 保存到配置文件
                self.config["model"] = new_model
                self.config.save_config()

                logger.info(f"[Gemini Image] 模型已从 {old_model} 切换到 {new_model}")
                yield event.plain_result(f"✅ 模型已切换: {new_model}")
            else:
                yield event.plain_result(f"❌ 无效的序号！请输入 1-{len(self.AVAILABLE_MODELS)} 之间的数字")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """预设管理指令

        用法:
        /预设 - 显示所有预设
        /预设 添加 <预设名:预设内容> - 添加新预设
        /预设 删除 <预设名> - 删除指定预设
        """
        # 从消息链中提取纯文本
        text_parts = []
        for seg in event.get_messages():
            if isinstance(seg, Comp.Plain):
                text_parts.append(seg.text)

        user_input = "".join(text_parts).strip()

        # 移除指令前缀
        possible_prefixes = []
        for wake_prefix in self.wake_prefixes:
            possible_prefixes.append(f"{wake_prefix}预设 ")
            possible_prefixes.append(f"{wake_prefix}预设")
        possible_prefixes.extend(["预设 ", "预设"])

        # 按长度降序排序，优先匹配更长的前缀
        possible_prefixes.sort(key=len, reverse=True)

        for prefix in possible_prefixes:
            if user_input.startswith(prefix):
                user_input = user_input[len(prefix):].strip()
                break

        # 如果没有参数，显示预设列表
        if not user_input:
            if not self.presets:
                yield event.plain_result("📋 当前没有预设\n\n💡 使用 /预设 添加 <预设名:预设内容> 来添加预设")
                return

            preset_list = "📋 预设列表:\n\n"
            for idx, (name, prompt) in enumerate(self.presets.items(), 1):
                # 截断过长的提示词
                display_prompt = prompt if len(prompt) <= 50 else prompt[:47] + "..."
                preset_list += f"{idx}. {name}: {display_prompt}\n"

            preset_list += "\n💡 使用方法:\n• /预设 添加 <预设名:预设内容>\n• /预设 删除 <预设名>"

            yield event.plain_result(preset_list)
            return

        # 处理"添加"子命令
        if user_input.startswith("添加 "):
            preset_str = user_input[3:].strip()

            if ":" not in preset_str:
                yield event.plain_result("❌ 格式错误！正确格式: /预设 添加 <预设名:预设内容>")
                return

            # 分割预设名和内容
            name, prompt = preset_str.split(":", 1)
            name = name.strip()
            prompt = prompt.strip()

            if not name or not prompt:
                yield event.plain_result("❌ 预设名和预设内容不能为空")
                return

            # 添加预设
            self.presets[name] = prompt

            # 保存到配置文件
            presets_config = [f"{k}:{v}" for k, v in self.presets.items()]
            self.config["presets"] = presets_config
            self.config.save_config()

            logger.info(f"[Gemini Image] 添加预设: {name}")
            yield event.plain_result(f"✅ 预设已添加: {name}")
            return

        # 处理"删除"子命令
        if user_input.startswith("删除 "):
            preset_name = user_input[3:].strip()

            if not preset_name:
                yield event.plain_result("❌ 请指定要删除的预设名")
                return

            if preset_name not in self.presets:
                yield event.plain_result(f"❌ 预设不存在: {preset_name}")
                return

            # 删除预设
            del self.presets[preset_name]

            # 保存到配置文件
            presets_config = [f"{k}:{v}" for k, v in self.presets.items()]
            self.config["presets"] = presets_config
            self.config.save_config()

            logger.info(f"[Gemini Image] 删除预设: {preset_name}")
            yield event.plain_result(f"✅ 预设已删除: {preset_name}")
            return

        # 未知子命令
        yield event.plain_result("❌ 未知命令\n\n💡 使用方法:\n• /预设 - 显示所有预设\n• /预设 添加 <预设名:预设内容>\n• /预设 删除 <预设名>")

    def _get_reply_message_chain(self, reply_component: Comp.Reply) -> list | None:
        """从 Reply 组件中获取被引用的消息链

        Args:
            reply_component: Reply 组件实例

        Returns:
            消息链列表，如果无法获取则返回 None
        """
        # 标准属性：chain
        if hasattr(reply_component, "chain") and isinstance(reply_component.chain, list):
            logger.debug("[Gemini Image] 使用标准属性 'chain' 获取引用消息")
            return reply_component.chain

        # 兼容性：尝试其他可能的属性名
        for attr_name in ["message", "source.message_chain"]:
            if "." in attr_name:
                # 处理嵌套属性访问
                parts = attr_name.split(".")
                obj = reply_component
                for part in parts:
                    if not hasattr(obj, part):
                        break
                    obj = getattr(obj, part)
                else:
                    if isinstance(obj, list):
                        logger.debug(f"[Gemini Image] 使用兼容属性 '{attr_name}' 获取引用消息")
                        return obj
            else:
                # 简单属性访问
                if hasattr(reply_component, attr_name):
                    value = getattr(reply_component, attr_name)
                    if isinstance(value, list):
                        logger.debug(f"[Gemini Image] 使用兼容属性 '{attr_name}' 获取引用消息")
                        return value

        logger.warning("[Gemini Image] 无法从 Reply 组件中获取消息链")
        return None

    async def _get_reference_images_for_tool(
        self, event: AstrMessageEvent, num_cached_images: int = 0
    ) -> list[tuple[bytes, str]]:
        """获取参考图片列表（用于工具调用）

        Args:
            event: 消息事件
            num_cached_images: 使用缓存图片的数量（当消息中没有图片时），0表示不使用缓存

        Returns:
            参考图片列表，每个元素为 (image_data, mime_type) 元组
        """
        images_data = []
        message_chain = event.message_obj.message

        # 首先处理引用消息中的图片
        for component in message_chain:
            if isinstance(component, Comp.Reply):
                logger.debug("[Gemini Image] 检测到引用消息，尝试解析被引用的图片")

                # 获取引用消息的消息链（标准属性是 chain）
                source_chain = self._get_reply_message_chain(component)

                # 从引用消息中提取所有图片（排除头像）
                if source_chain:
                    for replied_part in source_chain:
                        if isinstance(replied_part, Comp.Image) and hasattr(replied_part, "url") and replied_part.url:
                            if result := await self._download_image(replied_part.url):
                                images_data.append(result)
                                logger.debug("[Gemini Image] 成功从引用消息中加载图片")

                # 找到 Reply 组件后就跳出循环，通常一个消息链只有一个 Reply
                break

        # 继续处理当前消息中的图片
        for component in message_chain:
            if isinstance(component, Comp.Image):
                if result := await self._download_image(component.url or component.file):
                    images_data.append(result)

        # 如果消息中没有图片，且指定了缓存数量，则从缓存获取指定数量的最新图片
        if not images_data and num_cached_images > 0:
            recent_images = self.get_recent_images(event.unified_msg_origin)
            if recent_images:
                # 获取指定数量的最新图片
                for img_info in recent_images[:num_cached_images]:
                    if result := await self._download_image(img_info["url"]):
                        images_data.append(result)
                if images_data:
                    logger.debug(f"[Gemini Image] 从缓存中获取 {len(images_data)} 张参考图片")

        if images_data:
            logger.info(f"[Gemini Image] 共获取 {len(images_data)} 张参考图片")

        return images_data

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听消息，缓存用户发送的图片 URL"""
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                image_url = component.url or component.file
                if image_url:
                    self._remember_image_url(
                        event.unified_msg_origin, image_url, "image/png"
                    )

    def get_recent_images(self, session_id: str) -> list[dict]:
        """获取会话的最近图片"""
        # 先清理过期图片
        self._cleanup_expired_images(session_id)
        return self.recent_images.get(session_id, [])

    async def _periodic_cleanup_images(self):
        """定时清理过期图片的后台任务"""
        cleanup_interval = 600  # 每10分钟清理一次
        try:
            while True:
                await asyncio.sleep(cleanup_interval)
                self._cleanup_expired_images()
                logger.debug("[Gemini Image] 定时清理任务已执行")
        except asyncio.CancelledError:
            logger.debug("[Gemini Image] 定时清理任务已取消")
            raise

    def _cleanup_expired_images(self, session_id: str | None = None) -> None:
        """清理过期图片"""
        current_time = time.time()
        sessions = [session_id] if session_id else list(self.recent_images.keys())

        for sid in sessions:
            if sid not in self.recent_images:
                continue

            valid_images = [
                img for img in self.recent_images[sid]
                if current_time - img["timestamp"] < self.image_cache_ttl
            ]

            if valid_images:
                if len(valid_images) < len(self.recent_images[sid]):
                    logger.debug(f"[Gemini Image] 清理会话 {sid} 的 {len(self.recent_images[sid]) - len(valid_images)} 张过期图片")
                self.recent_images[sid] = valid_images
            else:
                del self.recent_images[sid]

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """统一创建后台任务并追踪生命周期"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def _get_download_session(self) -> aiohttp.ClientSession:
        """获取或创建用于下载图片的 aiohttp session"""
        if not hasattr(self, "_download_session") or self._download_session is None or self._download_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._download_session = aiohttp.ClientSession(timeout=timeout)
        return self._download_session

    async def _close_download_session(self):
        """关闭下载图片的 aiohttp session"""
        if hasattr(self, "_download_session") and self._download_session and not self._download_session.closed:
            await self._download_session.close()
            self._download_session = None

    @staticmethod
    async def get_avatar(user_id: str) -> bytes | None:
        """下载QQ用户头像

        Args:
            user_id: QQ用户ID

        Returns:
            头像数据，失败返回 None
        """
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with aiohttp.ClientSession() as client:
                response = await client.get(avatar_url)
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            logger.error(f"[Gemini Image] 下载头像失败: {e}")
            return None

    async def _download_image(self, image_url: str | None) -> tuple[bytes, str] | None:
        """下载图片并返回数据与 MIME 类型"""
        if not image_url:
            return None

        try:
            # 处理本地文件路径（file:// 协议）
            if image_url.startswith("file://"):
                file_path = image_url.removeprefix("file://")
                try:
                    # 使用 asyncio.to_thread 在线程池中读取文件，避免阻塞事件循环
                    def read_file():
                        with open(file_path, "rb") as f:
                            return f.read()

                    image_data = await asyncio.to_thread(read_file)

                    if len(image_data) > self.max_image_size:
                        logger.warning(f"[Gemini Image] 图片大小超过限制: {len(image_data)} > {self.max_image_size} bytes")
                        return None

                    # 根据文件扩展名推断 MIME 类型
                    import mimetypes
                    mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
                    logger.debug(f"[Gemini Image] 读取本地图片成功: {len(image_data)} bytes, MIME: {mime_type}")
                    return image_data, mime_type
                except FileNotFoundError:
                    logger.warning(f"[Gemini Image] 本地图片文件不存在: {file_path}")
                    return None
                except Exception as e:
                    logger.error(f"[Gemini Image] 读取本地图片失败: {e}")
                    return None

            # 处理 HTTP/HTTPS URL
            session = self._get_download_session()
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    logger.error(f"[Gemini Image] 下载图片失败: {resp.status} - {image_url}")
                    return None

                image_data = await resp.read()
                if len(image_data) > self.max_image_size:
                    logger.warning(f"[Gemini Image] 图片大小超过限制: {len(image_data)} > {self.max_image_size} bytes")
                    return None

                mime_type = resp.headers.get("Content-Type", "image/png")
                logger.debug(f"[Gemini Image] 下载图片成功: {len(image_data)} bytes, MIME: {mime_type}")
                return image_data, mime_type

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(f"[Gemini Image] 下载图片时出错: {exc}")
            return None

    def _remember_image_url(self, session_id: str, image_url: str, mime_type: str | None) -> None:
        """缓存图片 URL（而非完整数据，节省内存）

        Args:
            session_id: 会话ID
            image_url: 图片URL
            mime_type: MIME类型
        """
        session_images = self.recent_images.setdefault(session_id, [])
        session_images.insert(0, {
            "url": image_url,
            "mime_type": mime_type or "image/png",
            "timestamp": time.time(),
        })

        if len(session_images) > self.max_images_per_session:
            del session_images[self.max_images_per_session:]

        logger.debug(f"[Gemini Image] 已缓存图片 URL，会话 {session_id} 当前有 {len(session_images)} 张图片")

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
    ):
        """异步生成图片并发送给用户

        Args:
            prompt: 生成提示词
            unified_msg_origin: 消息来源
            images_data: 参考图片列表，每个元素为 (image_data, mime_type) 元组
            aspect_ratio: 宽高比
            resolution: 分辨率
        """
        task_id = hashlib.md5(f"{time.time()}{unified_msg_origin}".encode()).hexdigest()[:8]
        start_time = time.time()

        async with self._generation_semaphore:
            try:
                mode = "图生图" if images_data else "文生图"
                logger.info(f"[Gemini Image] [{task_id}] 开始{mode}任务，会话: {unified_msg_origin}")
                logger.debug(f"[Gemini Image] [{task_id}] 提示词: {prompt}")

                result_data, error = await self.generator.generate_image(
                    prompt=prompt,
                    images_data=images_data,
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                    task_id=task_id,
                )

                if error:
                    elapsed = time.time() - start_time
                    logger.warning(f"[Gemini Image] [{task_id}] {mode}任务失败，耗时: {elapsed:.2f}s")
                    await self._send_error_message(unified_msg_origin, error)
                    return

                image_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()
                file_path = await self.generator.cache_image(image_id, result_data)
                await self.context.send_message(
                    unified_msg_origin, MessageChain().file_image(str(file_path))
                )

                # 缓存 bot 生成的图片路径（使用 file:// 协议）
                file_url = f"file://{file_path.as_posix()}" if hasattr(file_path, "as_posix") else f"file://{file_path}"
                self._remember_image_url(unified_msg_origin, file_url, "image/png")

                elapsed = time.time() - start_time
                logger.info(f"[Gemini Image] [{task_id}] {mode}任务完成，耗时: {elapsed:.2f}s")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[Gemini Image] [{task_id}] 异步生成任务失败，耗时: {elapsed:.2f}s，错误: {e}", exc_info=True)
                await self._send_error_message(
                    unified_msg_origin, "图片生成过程中发生未知错误，请稍后重试或联系管理员"
                )

    async def _send_error_message(self, unified_msg_origin: str, error: str):
        """发送错误消息"""
        error_msg = f"❌ 图片生成失败: {error}"
        logger.error(f"[Gemini Image] {error_msg}")
        try:
            await self.context.send_message(unified_msg_origin, MessageChain().message(error_msg))
        except Exception:
            pass

    async def terminate(self):
        """插件卸载时清理资源"""
        try:
            logger.info("[Gemini Image] 开始卸载插件...")

            # 1. 先关闭网络连接（避免任务取消时还在使用）
            try:
                await self._close_download_session()
                logger.info("[Gemini Image] 已关闭下载 session")
            except Exception as e:
                logger.error(f"[Gemini Image] 关闭下载 session 失败: {e}")

            try:
                if hasattr(self, "generator") and self.generator:
                    await self.generator.close_session()
                    logger.info("[Gemini Image] 已关闭生成器 session")
            except Exception as e:
                logger.error(f"[Gemini Image] 关闭生成器 session 失败: {e}")

            # 2. 取消所有后台任务（包括定时清理任务）
            if hasattr(self, "background_tasks") and (pending_count := len(self.background_tasks)) > 0:
                logger.info(f"[Gemini Image] 正在取消 {pending_count} 个后台任务...")
                for task in self.background_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
                logger.info("[Gemini Image] 所有后台任务已取消")

            # 3. 清理内存缓存
            if hasattr(self, "recent_images"):
                total_images = sum(len(images) for images in self.recent_images.values())
                self.recent_images.clear()
                logger.info(f"[Gemini Image] 已清理内存中的图片缓存 ({total_images} 张)")

            # 4. 清理生成器缓存（包括磁盘文件）
            if hasattr(self, "generator") and self.generator and hasattr(self.generator, "image_cache"):
                cache_count = len(self.generator.image_cache)
                # 删除磁盘上的缓存文件
                deleted_files = 0
                for image_id in list(self.generator.image_cache.keys()):
                    try:
                        await self.generator._remove_cache(image_id)
                        deleted_files += 1
                    except Exception as e:
                        logger.warning(f"[Gemini Image] 删除缓存文件失败: {e}")
                logger.info(f"[Gemini Image] 已清理生成器缓存 ({cache_count} 个，删除 {deleted_files} 个文件)")

            logger.info("[Gemini Image] 插件已成功卸载")
        except Exception as e:
            logger.error(f"[Gemini Image] 清理资源时出错: {e}", exc_info=True)
