# Gemini Image 插件

![:name](https://count.getloli.com/@astrbot_plugin_gemini_image?name=astrbot_plugin_gemini_image&theme=miku&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

### 基于 Gemini 模型的图像生成插件，支持文生图和图生图，支持自然语言调用和预设提示词

***

### 命令
- `/生图 <提示词或预设名称>`
  - 生成图片，示例：`/生图 一只可爱的小猫`
  - 使用预设，示例：`/生图 手办化`
  - 如果消息中包含图片或引用包含图片的消息，则自动切换为图生图模式
  - 支持从最近缓存的图片中选择参考图（通过 LLM 工具调用）
  - 只发送`生图`时会返回帮助和`预设列表`

### 功能特性

- ✅ **文生图**：根据文字描述生成图片
- ✅ **图生图**：基于参考图片生成新图片（支持多张参考图）
- ✅ **预设提示词**：快速使用预定义的提示词模板
- ✅ **LLM 集成**：支持在对话中自动调用图像生成功能
- ✅ **智能缓存**：自动缓存最近的图片，支持基于时间和数量的清理
- ✅ **并发控制**：限制同时生成的图片数量，防止资源耗尽
- ✅ **多 API Key 轮询**：支持多个 API Key 自动切换，提高稳定性

### 配置项

| 配置项                       | 类型    | 默认值                                        | 说明                                          |
| ---------------------------- | ------- | --------------------------------------------- | --------------------------------------------- |
| `use_system_provider`        | boolean | `true`                                        | 是否使用系统提供商配置                        |
| `provider_id`                | string  | `""`                                          | 系统提供商ID                                  |
| `api_key`                    | list    | `[]`                                          | Gemini API Key 列表，支持多个 key 轮询        |
| `base_url`                   | string  | `"https://generativelanguage.googleapis.com"` | API基础URL                                    |
| `model`                      | string  | `"gemini-2.5-flash-image-preview"`            | 使用的模型名称                                |
| `custom_model`               | string  | `""`                                          | 自定义模型名称（当model为"自定义模型"时使用） |
| `timeout`                    | number  | `300`                                         | 生成超时时间（秒）                            |
| `cache_ttl`                  | number  | `3600`                                        | 图片缓存过期时间（秒）                        |
| `max_cache_count`            | number  | `50`                                          | 最大缓存数量                                  |
| `max_image_size_mb`          | number  | `10`                                          | 最大图片大小（MB）                            |
| `max_concurrent_generations` | number  | `3`                                           | 最大并发生成数                                |
| `max_requests_per_minute`    | number  | `3`                                           | 每分钟最大请求数                              |
| `enable_llm_tool`            | boolean | `true`                                        | 是否启用LLM工具集成                           |
| `default_aspect_ratio`       | string  | `"1:1"`                                       | /生图命令的默认宽高比                         |
| `default_resolution`         | string  | `"1K"`                                        | /生图命令的默认分辨率                         |
| `presets`                    | list    | `[...]`                                       | 预设提示词列表，格式：`"名称:提示词"`         |

### 预设提示词

预设提示词使用 `"名称:提示词"` 格式，例如：
```
"可爱小猫:a cute fluffy kitten with big eyes, sitting on a soft cushion, warm lighting, high quality, detailed fur"
```

用户可以在配置中添加自定义预设。


### 支持的模型
- `gemini-2.0-flash-exp-image-generation`
- `gemini-2.5-flash-image`
- `gemini-2.5-flash-image-preview` (默认)
- `gemini-3-pro-image-preview`
- `自定义模型`

### 使用示例

#### 基础用法
```
/生图 一只可爱的小猫
```

#### 使用预设
```
/生图 手办化
```

#### 图生图（引用消息）
```
[引用包含图片的消息]
/生图 手办化
```

#### LLM 自动调用
```
用户：帮我画一只小狗
LLM：好的，这就为您生成！
[自动调用图像生成工具]
```

### 更新日志

#### v1.0
- 正式版发布
- 优化日志输出

#### beta v1.8
- ✅ 添加预设提示词功能
- ✅ 指令名称改为 `/生图`
- ✅ 优化内存使用（ImageCache 不再存储完整图片数据）
- ✅ 实现基于时间的缓存清理
- ✅ 修复图片格式转换逻辑
- ✅ 添加图片下载超时控制
- ✅ 优化 LLM Tool 调用逻辑
- ✅ 清理死代码和冗余代码

#### beta v1.6
- 修复多个逻辑问题
- 优化错误处理
- 改进日志输出

#### beta v1.0
- 实现插件基本功能
- 支持文生图和图生图
- 添加图片缓存和并发控制
- 支持多种Gemini模型和配置选项