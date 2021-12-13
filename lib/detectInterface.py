from .detection import get_detector, get_textbox
from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, calculate_md5, get_paragraph,\
    download_and_unzip, diff, reformat_input,\
    make_rotated_img_list, set_result_with_confidence
from .config import *
from bidi.algorithm import get_display
import torch
import os
import sys
from logging import getLogger

from pathlib import Path

LOGGER = getLogger(__name__)


class Detector(object):

    def __init__(self, lang_list, gpu=True, model_storage_directory=None,
                 user_network_directory=None, recog_network='standard',
                 download_enabled=True, detector=True, recognizer=True,
                 verbose=True, quantize=True, cudnn_benchmark=False):
        self.download_enabled = download_enabled

        self.model_storage_directory = MODULE_PATH + '/model'
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        self.user_network_directory = MODULE_PATH + '/user_network'
        if user_network_directory:
            self.user_network_directory = user_network_directory
        Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        if gpu is False:
            self.device = 'cpu'
            if verbose:
                LOGGER.warning(
                    'Using CPU.')
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            if verbose:
                LOGGER.warning(
                    'CUDA not available - defaulting to CPU.')
        elif gpu is True:
            self.device = 'cuda'
        else:
            self.device = gpu
        self.recognition_models = recognition_models

        # check and download detection model
        detector_model = 'craft'
        corrupt_msg = 'MD5 hash mismatch, possible file corruption'
        detector_path = os.path.join(
            self.model_storage_directory, detection_models[detector_model]['filename'])

        # recognition model
        separator_list = {}

        # if recog_network in ['standard'] + [model for model in recognition_models['gen2']]:
        if lang_list == ['en']:
            self.set_language_model(
                'english', lang_list, ['en'], '["en"]')
            model = recognition_models['gen2']['english_g2']
            recog_network = 'generation2'
        self.character = model['characters']

        model_path = os.path.join(
            self.model_storage_directory, model['filename'])

        self.set_language_list(lang_list, model)

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        if detector:
            self.detector = get_detector(
                detector_path, self.device, quantize, cudnn_benchmark=cudnn_benchmark)
        if recognizer:
            network_params = {
                'input_channel': 1,
                'output_channel': 256,
                'hidden_size': 256
            }
            self.recognizer, self.converter = get_recognizer(recog_network, network_params,
                                                             self.character, separator_list,
                                                             dict_list, model_path, device=self.device, quantize=quantize)

    def set_language_model(self, language, lang_list, list_lang, list_lang_string):
        self.model_lang = language
        if set(lang_list) - set(list_lang) != set():
            if language == 'ch_tra' or language == 'ch_sim':
                language = 'chinese'
            raise ValueError(language.capitalize(
            ) + ' is only compatible with English, try lang_list=' + list_lang_string)

    def get_charactor(self, fileName):
        char_file = os.path.join(BASE_PATH, 'character', fileName)
        with open(char_file, "r", encoding="utf-8-sig") as input_file:
            list = input_file.read().splitlines()
            char = ''.join(list)
        return char

    def set_language_list(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(
                BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                char_list = input_file.read().splitlines()
            self.lang_char += char_list
        if model:
            symbol = model['symbols']
        else:
            symbol = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        self.lang_char = set(self.lang_char).union(set(symbol))
        self.lang_char = ''.join(self.lang_char)

    def detect(self, img, min_size=20, text_threshold=0.7, low_text=0.4,
               link_threshold=0.4, canvas_size=2560, mag_ratio=1.,
               slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
               width_ths=0.5, add_margin=0.1, reformat=True, optimal_num_chars=None):

        if reformat:
            img, _ = reformat_input(img)

        text_box_list = get_textbox(self.detector, img, canvas_size, mag_ratio,
                                    text_threshold, link_threshold, low_text,
                                    False, self.device, optimal_num_chars)

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,
                  decoder='greedy', beamWidth=5, batch_size=1,
                  workers=0, allowlist=None, blocklist=None, detail=1,
                  rotation_info=None, paragraph=False,
                  contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
                  y_ths=0.5, x_ths=1.0, reformat=True, output_format='standard'):

        if reformat:
            _, img_cv_grey = reformat_input(img_cv_grey)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))

        if self.model_lang in ['chinese_tra', 'chinese_sim']:
            decoder = 'greedy'

        if (horizontal_list == None) and (free_list == None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # without gpu/parallelization
        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(
                    h_list, f_list, img_cv_grey, model_height=imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,
                                   ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,
                                   workers, self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(
                    h_list, f_list, img_cv_grey, model_height=imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,
                                   ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,
                                   workers, self.device)
                result += result0
        # default mode
        else:
            image_list, max_width = get_image_list(
                horizontal_list, free_list, img_cv_grey, model_height=imgH)
            image_len = len(image_list)
            if rotation_info and image_list:
                image_list = make_rotated_img_list(rotation_info, image_list)
                max_width = max(max_width, imgH)

            result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,
                              workers, self.device)

            if rotation_info and (horizontal_list+free_list):
                result = set_result_with_confidence(result, image_len)

        if self.model_lang == 'arabic':
            direction_mode = 'rtl'
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
            direction_mode = 'ltr'

        if paragraph:
            result = get_paragraph(result, x_ths=x_ths,
                                   y_ths=y_ths, mode=direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            return [{'boxes': item[0], 'text':item[1], 'confident':item[2]} for item in result]
        else:
            return result

    def read(self, image, decoder='greedy', beamWidth=5, batch_size=1,
             workers=0, allowlist=None, blocklist=None, detail=1,
             rotation_info=None, paragraph=False, min_size=20,
             contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
             text_threshold=0.7, low_text=0.4, link_threshold=0.4,
             canvas_size=2560, mag_ratio=1.,
             slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
             width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, output_format='standard'):

        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, min_size, text_threshold,
                                                 low_text, link_threshold,
                                                 canvas_size, mag_ratio,
                                                 slope_ths, ycenter_ths,
                                                 height_ths, width_ths,
                                                 add_margin, False)
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(img_cv_grey, horizontal_list, free_list,
                                decoder, beamWidth, batch_size,
                                workers, allowlist, blocklist, detail, rotation_info,
                                paragraph, contrast_ths, adjust_contrast,
                                filter_ths, y_ths, x_ths, False, output_format)

        return result
