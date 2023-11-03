from Util import Util
import argparse
import warnings
import configparser
import ImageClass as Img

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--text', type = str)
    parser.add_argument('-g', '--generator', type = str)
    parser.add_argument('-d', '--dataset', type = str)
    parser.add_argument('-c', '--config', type = str, default = 'configs/mixor10-weight0.5-0.5-s.ini')
    
    
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    model = Util(args.generator, args.dataset, config = config)
    model.setup([args.text])

    imgs, xs, fs = model.get_fes()
    name = args.text.replace(' ', '_').replace('.', '')
    print(f"image name : {name}.png")
    Img.save(imgs[0], f'{name}.png')