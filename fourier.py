'''
由于OpenCV为了提高离散傅里叶变换速度采取了一些措施（可能包含补零操作等等），
因而在傅里叶变换过程中用numpy库取代cv2库。

注意: cv.cvtColor() 采用 np.uint8 或 np.float32 类型会产生误差，亦可能越界。

【numpy 常用操作】
a = np.repeat(ndarray[..., np.newaxis], 3, axis=-1)
a = np.swapaxes(ndarray, 0, 1).copy()
x, y = np.meshgrid(np.linspace(-0.5, 0.5, shape[1]), np.linspace(0.5, -0.5, shape[0]))
...

'''

import argparse
import os
import sys
import traceback
from datetime import datetime
from typing import List

import cv2 as cv
import numpy as np


class Fourier(object):
    color_channel = {
        'g': '灰 gray',
        'bgr': '蓝 blue, 绿 green, 红 red',
        'hls': '色相 hue, 亮度 lightness, 饱和度 saturation',
        'hsv': '色相 hue, 饱和度 saturation, 明度 value or brightness',
        'lab': '明度, 红绿, 黄蓝',
        'luv': '明度, 色度, 色度',
        'yrb': '明度, 色度, 色度',
    }
    color_space = ['g', 'bgr', 'hls', 'hsv', 'lab', 'luv', 'yrb']

    @staticmethod
    def load_image(image_path: str, read_mode=cv.IMREAD_COLOR) -> np.ndarray:
        '''

        Args:
            image_path:
            read_mode: cv::ImreadModes

        Returns:
            np.float32, 2D or 2D*3, [0, 255]
        '''
        return cv.imdecode(np.fromfile(image_path, dtype=np.uint8), read_mode).astype(np.float32)

    @staticmethod
    def save_image(out_png: str, in_data: np.ndarray) -> None:
        '''

        Args:
            out_png:
            in_data: 2D or 2D*3

        Returns:

        '''
        np.clip(in_data, 0, 255, out=in_data)
        cv.imencode('.png', in_data)[1].tofile(out_png)
        print('{} saved.'.format(out_png))

    @staticmethod
    def __copy_diff(write: np.ndarray, read: np.ndarray) -> np.ndarray:
        diff_index = write.astype(np.uint8) != read.astype(np.uint8)
        write[diff_index] = read[diff_index]
        return write

    @staticmethod
    def convolute(image: np.ndarray, kernel: np.ndarray, normalize=False) -> np.ndarray:
        '''

        Args:
            image: 2D or 2D*n, [0, 255]
            kernel: convolution kernel, 2D or 2D*n (the same as org), [-128, 127]
            normalize:

        Returns:
            2D or 2D*n (the same as org), [0, 255]
        '''
        assert np.all(image.shape >= kernel.shape)

        Fw_org = np.fft.fft2(image, axes=(0, 1))
        Fw_ker = np.fft.fft2(kernel, s=image.shape[:2], axes=(0, 1))
        Fw_new = Fw_org * Fw_ker
        ft_new = np.fft.ifft2(Fw_new, axes=(0, 1))
        ft_new = np.roll(ft_new, -np.array(kernel.shape[:2]) // 2, axis=(0, 1))
        ft_new = ft_new.real
        if normalize:
            ft_new -= np.min(ft_new, axis=(0, 1))
            ft_new *= 255.0 / np.max(ft_new, axis=(0, 1))
        else:
            lowest = 255.0 * np.sum(np.where(kernel < 0, kernel, 0.0), axis=(0, 1))
            highest = 255.0 * np.sum(np.where(kernel > 0, kernel, 0.0), axis=(0, 1))
            ft_new -= lowest
            ft_new *= 255.0 / (highest - lowest)
        return ft_new

    @staticmethod
    def cvt(in_bgr: np.ndarray, space: str) -> np.ndarray:
        '''

        Args:
            in_bgr: 2D*3, [0, 255]
            space: Fourier.color_space

        Returns:
            2D or 2D*3, [0, 255]
        '''
        color = {
            'g': cv.COLOR_BGR2GRAY,
            'hls': cv.COLOR_BGR2HLS,
            'hsv': cv.COLOR_BGR2HSV,
            'lab': cv.COLOR_BGR2Lab,
            'luv': cv.COLOR_BGR2Luv,
            'yrb': cv.COLOR_BGR2YCrCb,
        }

        if space == 'bgr':
            in_bgr *= 1.0
            return in_bgr

        in_bgr *= 1.0 / 255
        image_new = cv.cvtColor(in_bgr, color[space])

        if space in ('g', 'yrb'):
            image_new *= 255.0
        elif space in ('hls', 'hsv'):
            image_new[..., 0] *= 0.5
            image_new[..., 1] *= 255
            image_new[..., 2] *= 255
        elif space == 'lab':
            image_new[..., 0] *= 255.0 / 100
            image_new[..., 1] += 128
            image_new[..., 2] += 128
        elif space == 'luv':
            image_new[..., 0] *= 255.0 / 100
            image_new[..., 1] = (image_new[..., 1] + 134) * 255.0 / 354
            image_new[..., 2] = (image_new[..., 2] + 140) * 255.0 / 262

        np.clip(image_new, 0, 255.999, out=image_new)
        return image_new

    @staticmethod
    def icvt_with_org(cvt_in_bgr: np.ndarray, cvt_out: np.ndarray, space: str) -> np.ndarray:
        '''

        Args:
            cvt_in_bgr: bgr image, 2D*3, [0, 255]
            cvt_out: 2D or 2D*3, [0, 255]
            space: Fourier.color_space

        Returns:
            bgr image, 2D*3, [0, 255]
        '''
        color = {
            'g': cv.COLOR_GRAY2BGR,
            'hls': cv.COLOR_HLS2BGR,
            'hsv': cv.COLOR_HSV2BGR,
            'lab': cv.COLOR_Lab2BGR,
            'luv': cv.COLOR_Luv2BGR,
            'yrb': cv.COLOR_YCrCb2BGR,
        }

        cvt_org = Fourier.cvt(cvt_in_bgr, space)
        np.clip(cvt_out, 0, 255.999, out=cvt_out)
        cvt_org = Fourier.__copy_diff(cvt_org, cvt_out)

        if space == 'bgr':
            return cvt_org

        if space in ('g', 'yrb'):
            cvt_org *= 1.0 / 255
        elif space in ('hls', 'hsv'):
            cvt_org[..., 0] *= 2.0
            cvt_org[..., 1] *= 1.0 / 255
            cvt_org[..., 2] *= 1.0 / 255
        elif space == 'lab':
            cvt_org[..., 0] *= 100.0 / 255
            cvt_org[..., 1] -= 128
            cvt_org[..., 2] -= 128
        elif space == 'luv':
            cvt_org[..., 0] *= 100.0 / 255
            cvt_org[..., 1] = cvt_org[..., 1] * (354.0 / 255) - 134
            cvt_org[..., 2] = cvt_org[..., 2] * (262.0 / 255) - 140

        org_new = cv.cvtColor(cvt_org, color[space])
        org_new *= 255.0
        np.clip(org_new, 0, 255.999, out=org_new)
        return org_new

    @staticmethod
    def dft(image: np.ndarray) -> tuple:
        '''

        Args:
            image: 2D or 2D*n, [0, 255]

        Returns:
            2D or 2D*n
            magnitude: 幅度谱, amplitude spectrum, [0, 1]
            angle: 相位谱, phase spectrum, (-pi, pi]
        '''
        Fw = np.fft.fft2(image, axes=(0, 1))
        Fw_shift = np.fft.fftshift(Fw, axes=(0, 1))
        org_sum = np.sum(image, axis=(0, 1))
        # RuntimeWarning: divide by zero encountered in np.log(np.abs([0]))
        magnitude = np.log(np.abs(Fw_shift)) * (1.0 / np.log(org_sum))
        angle = np.angle(Fw_shift)
        return magnitude, angle

    @staticmethod
    def idft(magnitude: np.ndarray, angle: np.ndarray, dft_in_sum: np.ndarray) -> np.ndarray:
        '''

        Args:
            magnitude: 2D or 2D*n, 幅度谱, amplitude spectrum, [0, 1]
            angle: 2D or 2D*n, 相位谱, phase spectrum, (-pi, pi]
            dft_in_sum: n

        Returns:
            2D or 2D*n, about [0, 255]
        '''
        magnitude = np.exp(magnitude * np.log(dft_in_sum))
        Fw_shift = magnitude * np.exp(1j * angle)
        Fw = np.fft.ifftshift(Fw_shift, axes=(0, 1))
        origin = np.fft.ifft2(Fw, axes=(0, 1))
        origin = origin.real
        return origin

    @staticmethod
    def dft_to_image(image: np.ndarray) -> tuple:
        '''

        Args:
            image: 2D or 2D*n, [0, 255]

        Returns:
            2D or 2D*n
            magnitude: 幅度谱, amplitude spectrum, [0, 255]
            angle: 相位谱, phase spectrum, [0, 255]
        '''
        magnitude, angle = Fourier.dft(image)
        magnitude *= 255.0
        angle = (angle + np.pi) * (255.0 / (2 * np.pi))
        return magnitude, angle

    @staticmethod
    def idft_from_image(magnitude: np.ndarray, angle: np.ndarray, dft_in_sum: np.ndarray) -> np.ndarray:
        '''

        Args:
            magnitude: 幅度谱, amplitude spectrum, 2D or 2D*n, [0, 255]
            angle: 相位谱, phase spectrum, 2D or 2D*n, [0, 255]
            dft_in_sum: n

        Returns:
            2D or 2D*n, about [0, 255]
        '''
        magnitude *= 1.0 / 255
        angle = angle * (2 * np.pi / 255.0) - np.pi
        img = Fourier.idft(magnitude, angle, dft_in_sum)
        return img

    @staticmethod
    def idft_with_org(dft_in: np.ndarray, magnitude: np.ndarray = None, angle: np.ndarray = None) -> np.ndarray:
        '''

        Args:
            dft_in: 2D or 2D*n
            magnitude: 幅度谱, amplitude spectrum, 2D or 2D*n, [0, 255]
            angle: 相位谱, phase spectrum, 2D or 2D*n, [0, 255]

        Returns:
            2D or 2D*n, about [0, 255]
        '''
        assert magnitude is not None or angle is not None
        if magnitude is not None:
            assert dft_in.shape == magnitude.shape
        if angle is not None:
            assert dft_in.shape == angle.shape

        org_sum = np.sum(dft_in, axis=(0, 1))
        org_mag, org_ang = Fourier.dft_to_image(dft_in)
        if magnitude is not None:
            org_mag = Fourier.__copy_diff(org_mag, magnitude)
        if angle is not None:
            org_ang = Fourier.__copy_diff(org_ang, angle)
        org_new = Fourier.idft_from_image(org_mag, org_ang, org_sum)
        return org_new

    @staticmethod
    def save_convolution(space: str, kernel_path: str, path: list) -> None:
        '''

        Args:
            space: Fourier.color_space
            kernel_path:
            path:

        Returns:

        '''
        read_mode = cv.IMREAD_GRAYSCALE if space == 'g' else cv.IMREAD_COLOR
        ker_name = os.path.basename(kernel_path)
        ker_name = ker_name[: ker_name.rfind('.')]
        kernel = Fourier.load_image(kernel_path, read_mode)
        kernel -= 128.0
        for _path in path:
            org_img = Fourier.load_image(_path, read_mode)
            org_new = Fourier.convolute(org_img, kernel)
            save_name = '{}_{}_{}.png'.format(
                _path[: _path.rindex('.')], ker_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            )
            Fourier.save_image(save_name, org_new)

    @staticmethod
    def save_cvt(space: str, in_bgr: list) -> List[str]:
        '''

        Args:
            space: Fourier.color_space
            in_bgr: files' paths

        Returns:

        '''
        png_path = []
        for _path in in_bgr:
            org_img = Fourier.load_image(_path)
            cvted_img = Fourier.cvt(org_img, space)
            save_name = '{}_{}.png'.format(_path[: _path.rindex('.')], space.upper())
            Fourier.save_image(save_name, cvted_img)
            png_path.append(save_name)
        return png_path

    @staticmethod
    def save_icvt_with_org(space: str, cvt_in_bgr: str, cvt_out: str) -> None:
        '''

        Args:
            space: Fourier.color_space
            cvt_in_bgr:
            cvt_out:

        Returns:

        '''
        org_img = Fourier.load_image(cvt_in_bgr)
        read_mode = cv.IMREAD_GRAYSCALE if space == 'g' else cv.IMREAD_COLOR
        cvt_out_img = Fourier.load_image(cvt_out, read_mode)
        org_new = Fourier.icvt_with_org(org_img, cvt_out_img, space)
        save_name = '{}_{}.png'.format(cvt_in_bgr[: cvt_in_bgr.rindex('.')], datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
        Fourier.save_image(save_name, org_new)

    @staticmethod
    def save_fft(space: str, path: list) -> None:
        '''

        Args:
            space: Fourier.color_space
            path:

        Returns:

        '''
        read_mode = cv.IMREAD_GRAYSCALE if space == 'g' else cv.IMREAD_COLOR
        for _path in path:
            org_img = Fourier.load_image(_path, read_mode)
            magnitude, angle = Fourier.dft_to_image(org_img)

            save_name = _path[: _path.rindex('.')]
            for i, j in zip(('Mag', 'Ang'), (magnitude, angle)):
                Fourier.save_image('{}_{}.png'.format(save_name, i), j)

    @staticmethod
    def save_ifft_with_org(space: str, fft_in: str, mag_path: str, ang_path: str = None) -> str:
        '''

        Args:
            space: Fourier.color_space
            fft_in:
            mag_path:
            ang_path:

        Returns:

        '''
        read_mode = cv.IMREAD_GRAYSCALE if space == 'g' else cv.IMREAD_COLOR
        org_img = Fourier.load_image(fft_in, read_mode)
        mag_img = Fourier.load_image(mag_path, read_mode)
        ang_img = Fourier.load_image(ang_path, read_mode) if ang_path else None
        org_new = Fourier.idft_with_org(org_img, mag_img, ang_img)
        save_name = '{}_{}.png'.format(fft_in[: fft_in.rindex('.')], datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
        Fourier.save_image(save_name, org_new)
        return save_name

    @staticmethod
    def save_cvt_fft(space: str, in_bgr: list) -> None:
        png_path = Fourier.save_cvt(space, in_bgr)
        Fourier.save_fft(space, png_path)

    @staticmethod
    def save_ifft_icvt_with_org(space: str, cvt_in_bgr: str, fft_in: str, mag_path: str, ang_path: str = None) -> None:
        cvt_out = Fourier.save_ifft_with_org(space, fft_in, mag_path, ang_path)
        Fourier.save_icvt_with_org(space, cvt_in_bgr, cvt_out)


def get_files(directory: str) -> list:
    '''
    获取文件路径

    '''
    ret_files = []
    if os.path.isdir(directory):
        base_path = os.path.abspath(directory)
        for root, dirs, files in os.walk(base_path):
            ret_files.extend(
                list(map(lambda x: os.path.join(root, x), filter(lambda x: x.lower().endswith(('.jpg', '.png')), files)))
            )
        print('\n'.join(['{:4d}: {}'.format(i, j) for i, j in enumerate(ret_files)]))
    return ret_files


def get_info(directory: str) -> dict:
    '''
    带提示信息的参数向导

    '''
    args_ = []

    print('\n'.join(['{:2d}: {}'.format(i, j) for i, j in enumerate(Fourier.color_space)]))
    unsuccess = True
    while unsuccess:
        choose = input("choose color space index . | gray('enter')")
        try:
            args_.append('-c')
            args_.append(Fourier.color_space[int(choose)])
            unsuccess = False
        except:
            args_.pop()

    print()
    files = get_files(directory)
    print()

    key_list = [
        '--cvt-fft',
        '--ifft-icvt',
        '--cvt',
        '--fft',
        '--ifft',
        '--icvt',
        '--ker',
    ]
    prompt_list = [
        '1.cvt 2.fft ↪ in_bgr1, [in_bgr2...] ?',
        '1.ifft 2.icvt ↪ cvt_in_bgr, fft_in, mag_path, [ang_path] ?',
        'convert images. ↪ in_bgr1, [in_bgr2...] ?',
        'fft images. ↪ img1, [img2...] ?',
        'inverse fft image. ↪ fft_in, mag_path, [ang_path] ?',
        'inverse convert image. ↪ cvt_in_bgr, cvt_out ?',
        'convolute images by kernel. ↪ kernel, img1, [img2...] ?',
    ]
    for prompt, key in zip(prompt_list, key_list):
        choose = input(prompt + " ('y') | no('enter')")
        if choose == 'y':
            choose = input('input images split by space(indexs | paths):')

            ls = []
            for i in choose.split(' '):
                try:
                    ls.append(files[int(i)])
                except:
                    if os.path.isfile(i):
                        ls.append(i)

            args_.append(key)
            args_ += ls
            break

    return get_args(args_)


def get_args(ls: list = None) -> dict:
    parser = argparse.ArgumentParser(description='Fourier transform.')

    parser.add_argument("others", action="store", default=[], nargs='*', help="other parameters.")

    parser.add_argument("-i", "--info", action="store", const='.', nargs='?', help="input file path with prompt information.")
    parser.add_argument("-c", "--color", action="store", default='g', choices=Fourier.color_space, help="image color space.")

    parser.add_argument("--ker", action="store", nargs='+', help="convolute images by kernel. ↪ kernel, img1, [img2...]")
    parser.add_argument("--cvt", action="store", nargs='+', help="convert images. ↪ in_bgr1, [in_bgr2...]")
    parser.add_argument("--icvt", action="store", nargs='+', help="inverse convert image. ↪ cvt_in_bgr, cvt_out")
    parser.add_argument("--fft", action="store", nargs='+', help="fft images. ↪ img1, [img2...]")
    parser.add_argument("--ifft", action="store", nargs='+', help="inverse fft image. ↪ fft_in, mag_path, [ang_path]")
    parser.add_argument("--cvt-fft", action="store", nargs='+', help="1.cvt 2.fft ↪ in_bgr1, [in_bgr2...]")
    parser.add_argument(
        "--ifft-icvt", action="store", nargs='+', help="1.ifft 2.icvt ↪ cvt_in_bgr, fft_in, mag_path, [ang_path]"
    )

    args = parser.parse_args(ls)

    return vars(args)


def main():
    args = {
        'others': [],
        'info': '',
        'color': '',
        'ker': None,
        'cvt': None,
        'icvt': None,
        'fft': None,
        'ifft': None,
        'cvt_fft': None,
        'ifft_icvt': None,
    }
    args.update(get_args())
    print(sys.argv)
    print(args)
    print()

    if args['info']:
        args = get_info(args['info'])

    if args['ker']:
        Fourier.save_convolution(args['color'], args['ker'][0], args['ker'][1:])
    elif args['cvt']:
        Fourier.save_cvt(args['color'], args['cvt'])
    elif args['icvt']:
        Fourier.save_icvt_with_org(args['color'], *args['icvt'])
    elif args['fft']:
        Fourier.save_fft(args['color'], args['fft'])
    elif args['ifft']:
        Fourier.save_ifft_with_org(args['color'], *args['ifft'])
    elif args['cvt_fft']:
        Fourier.save_cvt_fft(args['color'], args['cvt_fft'])
    elif args['ifft_icvt']:
        Fourier.save_ifft_icvt_with_org(args['color'], *args['ifft_icvt'])


if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()

    print('OK.')
    sys.exit()
