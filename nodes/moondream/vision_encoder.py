import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
    ToTensor,
    Normalize,
)

class VisionEncoder:
    def __init__(self, model_path: str = "model") -> None:
        # Загружаем модель
        self.model = torch.jit.load(f"{model_path}/vision.pt").to(dtype=torch.float32)
        
        # Определяем последовательность трансформаций
        self.preprocess = Compose(
            [
                Resize(size=(384, 384), interpolation=InterpolationMode.BICUBIC),  # Изменение размера
                ToTensor(),  # Преобразуем изображение в тензор
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Нормализация
            ]
        )

    def __call__(self, image: Image) -> torch.Tensor:
        with torch.no_grad():
            # Преобразуем изображение и добавляем размерность батча
            image_vec = self.preprocess(image.convert("RGB")).unsqueeze(0)
            image_vec = image_vec[:, :, :-6, :-6]  # Убираем последние 6 пикселей
            image_vec = rearrange(
                image_vec, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14  # Изменяем размерности
            )

            return self.model(image_vec)  # Возвращаем результат работы модели
