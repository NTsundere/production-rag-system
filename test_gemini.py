from app.llm import get_llm

llm = get_llm()
response = llm.invoke("Скажи 'привет' одним словом.")
print("Ответ:", response.content)