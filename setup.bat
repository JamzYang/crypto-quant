@echo off
chcp 65001 >nul
echo ========================================
echo 数字货币量化交易学习环境配置
echo ========================================
echo.

echo [1/4] 检查Python版本...
python --version
if %errorlevel% neq 0 (
    echo ❌ 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)
echo.

echo [2/4] 创建虚拟环境...
if exist quant_env (
    echo ⚠️  虚拟环境已存在，跳过创建
) else (
    python -m venv quant_env
    echo ✅ 虚拟环境创建成功
)
echo.

echo [3/4] 激活虚拟环境并安装依赖...
call quant_env\Scripts\activate.bat
echo 当前Python环境：
where python
echo.

echo 安装核心依赖包...
pip install --upgrade pip
pip install numpy pandas matplotlib jupyter
pip install ccxt requests python-dotenv

echo.
echo [4/4] 验证安装...
python -c "import numpy; import pandas; import matplotlib; print('✅ 核心库安装成功')"

echo.
echo ========================================
echo 🎉 环境配置完成！
echo ========================================
echo.
echo 下一步：
echo 1. 激活虚拟环境：quant_env\Scripts\activate
echo 2. 运行示例代码：python 阶段0-数学预热代码示例.py
echo 3. 查看学习路线：学习路线图.md
echo.
pause
