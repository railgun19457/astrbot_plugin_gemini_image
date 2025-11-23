"""
Gemini Image Generation Module
封装 Gemini API 的图像生成功能，支持文生图和图生图
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import aiofiles
import aiohttp
from PIL import Image

from astrbot.api import logger


@dataclass
class ImageCache:
    """图片缓存信息"""

    file_path: Path
    timestamp: float
    data: bytes
    mime_type: str


class GeminiImageGenerator:
    """Gemini 图像生成器"""

    SUPPORTED_MODELS = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview",
    ]

    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        model: str,
        timeout: int = 120,
        cache_dir: Path | None = None,
        cache_ttl: int = 3600,
        max_cache_count: int = 100,
    ):
        """
        初始化 Gemini 图像生成器

        Args:
            api_keys: Gemini API Keys 列表
            base_url: API 基础 URL
            model: 使用的模型名称
            timeout: 生成超时时间（秒）
            cache_dir: 图片缓存目录
            cache_ttl: 缓存过期时间（秒）
            max_cache_count: 最大缓存数量
        """
        self.api_keys = api_keys if api_keys else []
        self.current_key_index = 0
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_dir = cache_dir or Path("data/temp/gemini_images")
        self.cache_ttl = cache_ttl
        self.max_cache_count = max_cache_count

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 图片缓存字典 {image_id: ImageCache}
        self.image_cache: dict[str, ImageCache] = {}

        # 清理任务追踪
        self._cleanup_task: asyncio.Task | None = None

    async def start_cleanup(self):
        """启动缓存清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_cache_loop())
            logger.info("[Gemini Image] 缓存清理任务已启动")

    async def stop_cleanup(self):
        """停止缓存清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("[Gemini Image] 缓存清理任务已停止")

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _rotate_api_key(self):
        """切换到下一个 API Key"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(
                f"[Gemini Image] 切换到下一个 API Key (索引: {self.current_key_index})"
            )

    def _convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """转换不支持的图片格式为 JPEG"""
        supported_formats = ["image/jpeg", "image/png", "image/webp"]
        if mime_type in supported_formats:
            return image_data, mime_type

        logger.info(f"[Gemini Image] 转换图片格式: {mime_type} -> image/jpeg")

        try:
            # 打开图片
            img = Image.open(BytesIO(image_data))

            # 处理透明图片
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background

            # 转换为 JPEG
            output = BytesIO()
            img.save(output, format="JPEG", quality=95)
            converted_data = output.getvalue()

            logger.info("[Gemini Image] 图片格式转换成功")
            return converted_data, "image/jpeg"

        except Exception as e:
            logger.error(f"[Gemini Image] 图片格式转换失败: {e}")
            return image_data, mime_type

    async def generate_image(
        self,
        prompt: str,
        image_data: bytes | None = None,
        mime_type: str | None = None,
        aspect_ratio: str = "1:1",
        image_size: str | None = None,
    ) -> tuple[bytes | None, str | None]:
        """
        统一的图像生成接口，支持文生图和图生图

        Args:
            prompt: 生成图片的文本提示词
            image_data: 可选的参考图片字节数据，为 None 时执行文生图，否则执行图生图
            mime_type: 参考图片的 MIME 类型
            aspect_ratio: 宽高比，支持 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
            image_size: 图片分辨率（仅 Gemini 3 Pro），支持 1K, 2K, 4K

        Returns:
            (图片字节数据, 错误信息)
        """
        if not self.api_keys:
            return None, "未配置 API Key"

        # 如果有参考图片，转换格式（如果需要）
        if image_data and mime_type:
            image_data, mime_type = self._convert_image_format(image_data, mime_type)

        # 尝试所有可用的 API Key
        last_error = None
        for attempt in range(len(self.api_keys)):
            api_key = self._get_current_api_key()

            try:
                url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                }

                # 构建配置
                generation_config = {
                    "response_modalities": ["IMAGE"],
                }

                # 添加图片配置
                image_config = {}
                if aspect_ratio:
                    image_config["aspect_ratio"] = aspect_ratio
                if image_size:
                    image_config["image_size"] = image_size
                if image_config:
                    generation_config["image_config"] = image_config

                # 构建 parts
                parts = [{"text": prompt}]

                # 如果有参考图片，添加到 parts
                if image_data and mime_type:
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64,
                            }
                        }
                    )

                payload = {
                    "contents": [{"parts": parts}],
                    "generation_config": generation_config,
                }

                mode = "图生图" if image_data else "文生图"
                logger.info(
                    f"[Gemini Image] 开始{mode}，提示词: {prompt[:50]}... (Key 索引: {self.current_key_index})"
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"[Gemini Image] API 错误: {response.status} - {error_text}"
                            )
                            last_error = f"API 请求失败: {response.status}"

                            # 如果是认证错误或配额错误，尝试下一个 key
                            if (
                                response.status in [401, 403, 429]
                                and len(self.api_keys) > 1
                            ):
                                logger.warning(
                                    f"[Gemini Image] Key 失败，尝试下一个 (错误: {response.status})"
                                )
                                self._rotate_api_key()
                                continue

                            return None, last_error

                        result = await response.json()

                        # 解析响应获取图片数据
                        result_image_data = self._extract_image_from_response(result)
                        if result_image_data:
                            logger.info(f"[Gemini Image] {mode}成功")
                            return result_image_data, None
                        else:
                            logger.error("[Gemini Image] 响应中未找到图片数据")
                            return None, "响应中未找到图片数据"

            except asyncio.TimeoutError:
                logger.error("[Gemini Image] 生成超时")
                last_error = "生成超时"
                # 超时也尝试下一个 key
                if len(self.api_keys) > 1 and attempt < len(self.api_keys) - 1:
                    self._rotate_api_key()
                    continue
            except Exception as e:
                logger.error(f"[Gemini Image] 生成失败: {e}")
                last_error = f"生成失败: {str(e)}"
                # 其他错误也尝试下一个 key
                if len(self.api_keys) > 1 and attempt < len(self.api_keys) - 1:
                    self._rotate_api_key()
                    continue

        return None, last_error or "所有 API Key 都失败了"

    def _extract_image_from_response(self, response: dict) -> bytes | None:
        """从 API 响应中提取图片数据

        根据官方文档，响应格式为：
        {
            "candidates": [{
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": "<base64_image_data>"
                            },
                            "thought": true  // 可选，标记为思考过程图片
                        }
                    ]
                }
            }]
        }

        对于 Gemini 3 Pro Image Preview 模型：
        - 会生成多张临时图片（思考过程），这些图片带有 "thought": true
        - 最终渲染的图片是最后一张不带 "thought" 标志的图片
        """
        try:
            # 记录响应结构信息，避免输出完整的base64数据
            candidates_count = len(response.get("candidates", []))
            logger.debug(f"[Gemini Image] 解析响应: 找到 {candidates_count} 个候选结果")

            candidates = response.get("candidates", [])
            if not candidates:
                logger.warning("[Gemini Image] 响应中没有 candidates")
                return None

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            logger.debug(f"[Gemini Image] 找到 {len(parts)} 个 parts")

            # 收集所有非思考过程的图片
            final_images = []
            for i, part in enumerate(parts):
                # 检查是否为思考过程图片
                is_thought = part.get("thought", False)

                if is_thought:
                    logger.debug(f"[Gemini Image] 跳过思考过程图片 part {i}")
                    continue

                # 统一处理 inline_data 格式
                inline_data = part.get("inline_data") or part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mime_type") or inline_data.get(
                        "mimeType", ""
                    )

                    if mime_type.startswith("image/"):
                        image_base64 = inline_data.get("data", "")
                        if image_base64:
                            logger.debug(
                                f"[Gemini Image] 找到图片数据，长度: {len(image_base64)} 字符"
                            )
                            final_images.append(base64.b64decode(image_base64))
                        else:
                            logger.warning(f"[Gemini Image] part {i} 没有图片数据")
                    else:
                        logger.debug(f"[Gemini Image] 跳过非图片类型 part {i}")
                else:
                    logger.debug(f"[Gemini Image] part {i} 没有 inline_data")

            result = final_images[-1] if final_images else None
            logger.debug(
                f"[Gemini Image] 返回 {len(final_images)} 张图片中的最后一张"
                if final_images
                else "[Gemini Image] 未找到任何图片"
            )
            return result

        except Exception as e:
            logger.error(f"[Gemini Image] 解析响应失败: {e}")
            return None

    async def cache_image(
        self, image_id: str, image_data: bytes, mime_type: str = "image/jpeg"
    ) -> Path:
        """
        缓存图片到本地

        Args:
            image_id: 图片唯一标识
            image_data: 图片字节数据
            mime_type: 图片 MIME 类型

        Returns:
            缓存文件路径
        """
        # 获取文件扩展名
        ext = mimetypes.guess_extension(mime_type) or ".jpg"
        file_path = self.cache_dir / f"{image_id}{ext}"

        # 保存到文件
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(image_data)

        # 添加到缓存
        self.image_cache[image_id] = ImageCache(
            file_path=file_path,
            timestamp=time.time(),
            data=image_data,
            mime_type=mime_type,
        )

        # 检查缓存数量
        await self._check_cache_limit()

        logger.info(f"[Gemini Image] 图片已缓存: {file_path}")
        return file_path

    async def get_cached_image(self, image_id: str) -> tuple[bytes, str] | None:
        """
        获取缓存的图片

        Args:
            image_id: 图片唯一标识

        Returns:
            (图片字节数据, MIME 类型) 或 None
        """
        cache = self.image_cache.get(image_id)
        if not cache:
            return None

        # 检查是否过期
        if time.time() - cache.timestamp > self.cache_ttl:
            await self._remove_cache(image_id)
            return None

        # 从文件读取图片数据
        try:
            async with aiofiles.open(cache.file_path, "rb") as f:
                image_data = await f.read()
            # 从文件扩展名推断 MIME 类型
            mime_type = mimetypes.guess_type(cache.file_path)[0] or "image/jpeg"
            return image_data, mime_type
        except Exception as e:
            logger.error(f"[Gemini Image] 读取缓存文件失败: {e}")
            await self._remove_cache(image_id)
            return None

    async def _check_cache_limit(self):
        """检查并清理超出限制的缓存"""
        if len(self.image_cache) <= self.max_cache_count:
            return

        # 按时间戳排序，删除最旧的
        sorted_cache = sorted(self.image_cache.items(), key=lambda x: x[1].timestamp)

        to_remove = len(self.image_cache) - self.max_cache_count
        for image_id, _ in sorted_cache[:to_remove]:
            await self._remove_cache(image_id)

    async def _remove_cache(self, image_id: str):
        """删除缓存"""
        cache = self.image_cache.pop(image_id, None)
        if cache and cache.file_path.exists():
            try:
                cache.file_path.unlink()
            except Exception as e:
                logger.error(f"[Gemini Image] 删除缓存文件失败: {e}")

    async def _cleanup_cache_loop(self):
        """定期清理过期缓存"""
        while True:
            try:
                await asyncio.sleep(300)  # 每 5 分钟检查一次

                current_time = time.time()
                expired_ids = [
                    image_id
                    for image_id, cache in self.image_cache.items()
                    if current_time - cache.timestamp > self.cache_ttl
                ]

                for image_id in expired_ids:
                    await self._remove_cache(image_id)

                if expired_ids:
                    logger.info(f"[Gemini Image] 清理了 {len(expired_ids)} 个过期缓存")

            except Exception as e:
                logger.error(f"[Gemini Image] 缓存清理失败: {e}")
