from setuptools import setup, find_packages

setup(
    name="iroh-tandem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "motor>=3.3.1",
        "pymongo>=4.6.1",
        "python-dotenv>=1.0.0",
        "nvidia-ml-py3>=7.352.0",
        "psutil>=5.9.6",
        "pydantic>=2.4.2",
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "httpx>=0.25.1",
        "pytest-cov>=4.1.0",
        "iroh>=0.1.0",
        "aiohttp>=3.9.1",
        "pynvml>=11.5.0"
    ],
    python_requires=">=3.8",
) 