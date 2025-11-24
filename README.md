# Gemini Image 插件

![:name](https://count.getloli.com/@astrbot_plugin_gemini_image?name=astrbot_plugin_gemini_image&theme=miku&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

### 基于 Gemini 模型的图像生成插件，支持文生图和图生图，支持自然语言调用

***

### 命令
(命令组：`/img`)

- `/img <提示词>`
  - 生成图片，示例：`/img 一只可爱的小猫`
  - 如果消息中包含图片，则自动切换为图生图模式

### 配置项

| 配置项                       | 类型         | 默认值                                        | 说明                                      |
| ---------------------------- | ------------ | --------------------------------------------- | ----------------------------------------- |
| `use_system_provider`        | boolean      | `true`                                        | 是否使用系统提供商配置                    |
| `provider_id`                | string       | `""`                                          | 系统提供商ID                              |
| `api_key`                    | string/array | `""`                                          | Gemini API Key，支持单个key或数组         |
| `base_url`                   | string       | `"https://generativelanguage.googleapis.com"` | API基础URL                                |
| `model`                      | string       | `"gemini-2.0-flash-exp-image-generation"`     | 使用的模型名称                            |
| `custom_model`               | string       | `""`                                          | 自定义模型名称（当model为"custom"时使用） |
| `timeout`                    | number       | `120`                                         | 生成超时时间（秒）                        |
| `cache_ttl`                  | number       | `3600`                                        | 图片缓存过期时间（秒）                    |
| `max_cache_count`            | number       | `100`                                         | 最大缓存数量                              |
| `max_image_size_mb`          | number       | `10`                                          | 最大图片大小（MB）                        |
| `max_concurrent_generations` | number       | `3`                                           | 最大并发生成数                            |
| `max_requests_per_minute`    | number       | `5`                                           | 每分钟最大请求数                          |
| `enable_llm_tool`            | boolean      | `true`                                        | 是否启用LLM工具集成                       |
| `default_aspect_ratio`       | string       | `"1:1"`                                       | /img命令的默认宽高比                       |
| `default_resolution`         | string       | `"1K"`                                        | /img命令的默认分辨率                       |

### 支持的模型
- `gemini-2.0-flash-exp-image-generation`
- `gemini-2.5-flash-image`
- `gemini-2.5-flash-image-preview`
- `gemini-3-pro-image-preview`
- `自定义模型`

### 更新日志
#### v1.0
- 实现插件基本功能
- 支持文生图和图生图
- 添加图片缓存和并发控制
- 支持多种Gemini模型和配置选项