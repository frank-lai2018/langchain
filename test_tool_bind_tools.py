# 配置模型
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from rich import print as rprint

# 加载配置文件
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# 確保環境變數有 GOOGLE_API_KEY，或是直接在這裡把 key 傳進 api_key 參數
model = init_chat_model(
    model="gemma-4-31b-it",
    model_provider="openai",  # 關鍵：強制使用 OpenAI 的連線邏輯
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" # 指向 Google
)

# 定義工具
def get_weather(city : str):
    return f"{city}天氣真好，天氣晴!!"

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke('台北市')

rprint(response)