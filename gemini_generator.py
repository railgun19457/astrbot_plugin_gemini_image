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
        max_retry_attempts: int = 3,
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
            max_retry_attempts: 生成失败时的最大重试次数
        """
        self.api_keys = api_keys if api_keys else []
        self.current_key_index = 0
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_dir = cache_dir or Path("data/temp/gemini_images")
        self.cache_ttl = cache_ttl
        self.max_cache_count = max_cache_count
        self.max_retry_attempts = max(1, min(max_retry_attempts, 10))  # 限制在 1-10 次之间

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 图片缓存字典 {image_id: ImageCache}
        self.image_cache: dict[str, ImageCache] = {}

        # 复用的 aiohttp ClientSession
        self._session: aiohttp.ClientSession | None = None

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

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        """关闭 aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _sync_convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """同步的图片格式转换逻辑（在线程池中执行）"""
        try:
            # 打开图片
            img = Image.open(BytesIO(image_data))

            # 处理透明图片
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                elif img.mode == "LA":
                    img = img.convert("RGBA")

                # 此时 img.mode 一定是 RGBA，使用第4个通道作为 alpha mask
                background.paste(img, mask=img.split()[3])
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

    async def _convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """转换不支持的图片格式为 JPEG（异步，避免阻塞事件循环）"""
        supported_formats = ["image/jpeg", "image/png", "image/webp"]
        if mime_type in supported_formats:
            return image_data, mime_type

        logger.info(f"[Gemini Image] 转换图片格式: {mime_type} -> image/jpeg")

        # 使用 asyncio.to_thread 在线程池中执行同步的图片转换操作
        return await asyncio.to_thread(self._sync_convert_image_format, image_data, mime_type)

    async def generate_image(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        image_size: str | None = None,
        task_id: str | None = None,
    ) -> tuple[bytes | None, str | None]:
        """统一的图像生成接口，支持文生图和图生图（支持多张参考图片），带重试机制

        Args:
            prompt: 生成提示词
            images_data: 参考图片列表，每个元素为 (image_data, mime_type) 元组
            aspect_ratio: 宽高比
            image_size: 图片尺寸
            task_id: 任务ID，用于日志追踪

        Returns:
            (生成的图片数据, 错误信息) 元组
        """
        if not self.api_keys:
            return None, "未配置 API Key"

        prefix = f"[{task_id}] " if task_id else ""

        # 转换所有图片格式（如果需要）
        converted_images = []
        if images_data:
            for image_data, mime_type in images_data:
                converted_data, converted_mime = await self._convert_image_format(image_data, mime_type)
                converted_images.append((converted_data, converted_mime))

        # 重试机制：对每个 API Key 进行多次重试
        last_error = "未配置 API Key"

        for retry_attempt in range(self.max_retry_attempts):
            # 尝试所有可用的 API Key
            for key_attempt in range(len(self.api_keys)):
                if retry_attempt > 0 or key_attempt > 0:
                    logger.info(
                        f"[Gemini Image] {prefix}重试生成 (第 {retry_attempt + 1}/{self.max_retry_attempts} 次重试, "
                        f"Key 索引: {self.current_key_index})"
                    )

                result = await self._try_generate_with_current_key(
                    prompt, converted_images, aspect_ratio, image_size, task_id
                )

                if result[0] is not None:  # 成功
                    if retry_attempt > 0:
                        logger.info(f"[Gemini Image] {prefix}重试成功！")
                    return result

                last_error = result[1] or "生成失败"

                # 如果有多个 key 且不是最后一次尝试，切换到下一个 key
                if len(self.api_keys) > 1 and key_attempt < len(self.api_keys) - 1:
                    self._rotate_api_key()

            # 如果不是最后一次重试，等待一小段时间后再重试
            if retry_attempt < self.max_retry_attempts - 1:
                wait_time = min(2 ** retry_attempt, 10)  # 指数退避，最多等待10秒
                logger.info(f"[Gemini Image] {prefix}等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)

        logger.error(
            f"[Gemini Image] {prefix}所有重试均失败 "
            f"(共 {self.max_retry_attempts} 次重试 × {len(self.api_keys)} 个 Key)"
        )
        return None, f"重试 {self.max_retry_attempts} 次后仍然失败: {last_error}"

    async def _try_generate_with_current_key(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str,
        image_size: str | None,
        task_id: str | None = None,
    ) -> tuple[bytes | None, str | None]:
        """使用当前 API Key 尝试生成图片"""
        prefix = f"[{task_id}] " if task_id else ""
        start_time = time.time()

        try:
            payload = self._build_request_payload(
                prompt, images_data, aspect_ratio, image_size
            )

            mode = f"图生图({len(images_data)}张参考图)" if images_data else "文生图"
            logger.debug(
                f"[Gemini Image] {prefix}开始{mode} (Key 索引: {self.current_key_index})"
            )

            # 使用复用的 session
            session = self._get_session()
            response_data = await self._make_api_request(session, payload, task_id)
            if response_data is None:
                return None, "API 请求失败"

            # 解析响应获取图片数据
            result_image_data = self._extract_image_from_response(response_data, task_id)
            if result_image_data:
                elapsed = time.time() - start_time
                logger.debug(f"[Gemini Image] {prefix}API 请求成功，耗时: {elapsed:.2f}s")
                return result_image_data, None

            logger.error(f"[Gemini Image] {prefix}响应中未找到图片数据")
            return None, "响应中未找到图片数据"

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[Gemini Image] {prefix}生成超时，耗时: {elapsed:.2f}s")
            return None, "生成超时"
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Gemini Image] {prefix}生成失败，耗时: {elapsed:.2f}s，错误: {e}")
            return None, f"生成失败: {str(e)}"

    def _build_request_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str,
        image_size: str | None,
    ) -> dict:
        """构建 API 请求 payload（支持多张参考图片）

        Args:
            prompt: 生成提示词
            images_data: 参考图片列表，每个元素为 (image_data, mime_type) 元组
            aspect_ratio: 宽高比
            image_size: 图片尺寸

        Returns:
            API 请求 payload
        """
        # 只返回图片，不返回文本
        generation_config = {"responseModalities": ["IMAGE"]}

        # 添加图片配置
        image_config = {}
        if aspect_ratio:
            image_config["aspectRatio"] = aspect_ratio

        # imageSize 仅 gemini-3-pro-image-preview 支持
        if image_size and "gemini-3" in self.model.lower():
            image_config["imageSize"] = image_size

        if image_config:
            generation_config["imageConfig"] = image_config

        # 构建 parts - 先添加提示词
        parts = [{"text": prompt}]

        # 添加所有参考图片
        if images_data:
            for idx, (image_data, mime_type) in enumerate(images_data):
                encoded_data = base64.b64encode(image_data).decode("utf-8")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_data,
                        }
                    }
                )
                logger.debug(
                    f"[Gemini Image] 添加参考图 {idx + 1}/{len(images_data)}: "
                    f"大小={len(image_data)} bytes, MIME={mime_type}, "
                    f"Base64长度={len(encoded_data)} chars"
                )

        return {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        }

    async def _make_api_request(
        self, session: aiohttp.ClientSession, payload: dict, task_id: str | None = None
    ) -> dict | None:
        """发送 API 请求并处理响应"""
        prefix = f"[{task_id}] " if task_id else ""
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._get_current_api_key(),
        }

        logger.debug(f"[Gemini Image] {prefix}请求 URL: {url}")

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                # 截断过长的错误信息
                error_preview = error_text[:500] + "..." if len(error_text) > 500 else error_text
                logger.error(f"[Gemini Image] {prefix}API 错误: {response.status} - {error_preview}")

                # 如果是认证错误或配额错误，尝试下一个 key
                if response.status in [401, 403, 429] and len(self.api_keys) > 1:
                    logger.warning(
                        f"[Gemini Image] {prefix}Key 失败，尝试下一个 (错误: {response.status})"
                    )
                return None

            return await response.json()

    def _extract_image_from_response(self, response: dict, task_id: str | None = None) -> bytes | None:
        """从 API 响应中提取图片数据"""
        prefix = f"[{task_id}] " if task_id else ""
        try:
            candidates = response.get("candidates", [])
            logger.debug(f"[Gemini Image] {prefix}解析响应: 找到 {len(candidates)} 个候选结果")

            if not candidates:
                logger.warning(f"[Gemini Image] {prefix}响应中没有 candidates")
                logger.debug(f"[Gemini Image] {prefix}完整响应: {response}")
                return None

            candidate = candidates[0]
            logger.debug(f"[Gemini Image] {prefix}候选结果结构: {list(candidate.keys())}")

            # 检查是否有 content 字段
            if "content" not in candidate:
                logger.warning(f"[Gemini Image] {prefix}候选结果中没有 content 字段")
                logger.debug(f"[Gemini Image] {prefix}候选结果内容: {candidate}")
                return None

            content = candidate.get("content", {})
            parts = content.get("parts", [])
            logger.debug(f"[Gemini Image] {prefix}找到 {len(parts)} 个 parts")

            if not parts:
                logger.warning(f"[Gemini Image] {prefix}content 中没有 parts")
                logger.debug(f"[Gemini Image] {prefix}content 内容: {content}")
                return None

            # 收集所有图片
            images = []
            for i, part in enumerate(parts):
                image_data = self._extract_image_from_part(part, i, task_id)
                if image_data:
                    images.append(image_data)

            if images:
                logger.debug(f"[Gemini Image] {prefix}返回 {len(images)} 张图片中的最后一张")
                return images[-1]

            logger.debug(f"[Gemini Image] {prefix}未找到任何图片")
            return None

        except Exception as e:
            logger.error(f"[Gemini Image] {prefix}解析响应失败: {e}")
            return None

    def _extract_image_from_part(self, part: dict, index: int = 0, task_id: str | None = None) -> bytes | None:
        """从单个 part 中提取图片数据"""
        prefix = f"[{task_id}] " if task_id else ""
        inline_data = part.get("inline_data") or part.get("inlineData")
        if not inline_data:
            logger.debug(f"[Gemini Image] {prefix}part {index} 没有 inline_data")
            return None

        mime_type = inline_data.get("mime_type") or inline_data.get("mimeType", "")
        if not mime_type.startswith("image/"):
            logger.debug(f"[Gemini Image] {prefix}跳过非图片类型 part {index}")
            return None

        image_base64 = inline_data.get("data", "")
        if not image_base64:
            logger.warning(f"[Gemini Image] {prefix}part {index} 没有图片数据")
            return None

        logger.debug(f"[Gemini Image] {prefix}找到图片数据，长度: {len(image_base64)} 字符")
        return base64.b64decode(image_base64)

    async def cache_image(
        self, image_id: str, image_data: bytes, mime_type: str = "image/png"
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
        ext = mimetypes.guess_extension(mime_type) or ".png"
        file_path = self.cache_dir / f"{image_id}{ext}"

        # 保存到文件
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(image_data)

        # 添加到缓存
        self.image_cache[image_id] = ImageCache(
            file_path=file_path,
            timestamp=time.time(),
            mime_type=mime_type,
        )

        # 检查缓存数量
        await self._check_cache_limit()

        logger.debug(f"[Gemini Image] 图片已缓存: {file_path}")
        return file_path

    async def _check_cache_limit(self):
        """检查并清理超出限制的缓存（基于数量和时间）"""
        current_time = time.time()

        # 首先清理过期的缓存
        expired_ids = [
            image_id for image_id, cache in self.image_cache.items()
            if current_time - cache.timestamp > self.cache_ttl
        ]

        for image_id in expired_ids:
            await self._remove_cache(image_id)
            logger.debug(f"[Gemini Image] 清理过期缓存: {image_id}")

        # 如果清理后仍超出数量限制，删除最旧的
        if len(self.image_cache) > self.max_cache_count:
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
