# VPP Cloud Trainer

## 项目概述
VPP Cloud Trainer 是一个用于训练电力负荷的云服务项目。

## 功能说明
- **数据预测**：基于历史数据训练预测模型。
- **API 服务**：提供 RESTful API 接口供外部调用。

## 使用方法
1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **启动服务**：
   ```bash
   python main.py
   ```

## API 说明
API采用GET方式，是为了方便在浏览器中直接调用该API生成数据。更好的设计方式是采用POST方式，更符合RESTful API的设计规范。

### 1. 生成历史数据
- **请求方式** GET
- **请求地址**：`/api/v1/download`
```bash
   /api/v1/download # 将配置文件中指定的ftp地址的download_data_dir目录下所有文件下载到本地data.dir下
```
