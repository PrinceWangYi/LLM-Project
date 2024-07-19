from pydantic import BaseModel


# Pydantic是一个Python库，用于数据验证和设置管理。它被广泛用于FastAPI中，
# 用于定义请求和响应模型，以进行数据验证和解析。

class Item(BaseModel):
    name: str
    description: str
    price: float


items = {
    "foo": {"name": "foo", "description": "The foo", "price": 50.2},
    "bar": {"name": "bar", "description": "The bar", "price": 62.0}
}

item = Item(**items["bar"])
print(item)
