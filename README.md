# fourier-image
利用傅里叶变换去除图像中的一些规律噪声

## 简介

对原始图像进行色彩空间转换，利用图像编辑软件分通道编辑幅度图像，去除有规律的噪声（如网纹噪声等），然后进行傅里叶逆变换即可。

在傅里叶逆变换中，为了尽可能精确地接近原始图像，需要先对原始图像进行正变换，只替换幅度图像修改部分的数据。计算过程采用浮点数。

## 用法举例

```bash
# 帮助说明
python fourier.py -h

# 色彩空间转换，傅里叶变换
python fourier.py -c hsv --cvt-fft 02.jpg
# 编辑 02_HSV_Mag.png 如 02_HSV_Mag-fix.png 所示
# 傅里叶逆变换，色彩空间转换
python fourier.py -c hsv --ifft-icvt 02.jpg 02_HSV.png 02_HSV_Mag-fix.png

# 图像卷积
python fourier.py -c bgr --ker kernel.png 02.jpg

# 交互式用法
python fourier.py -i
```

