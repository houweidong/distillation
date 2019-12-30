from enum import Enum
import numpy as np

# Enum class for attributes included in Wider Attribute dataset only
class WiderAttributes(Enum):
    # { 0”：“男性”，“1”：“长发”，“2”：“太阳镜”“3”：“帽子”，“4”：“T-shirt”，“5”：“长袖”，“6”：“正装”,
    # “7”：“短裤”，“8”：“牛仔裤”“9”：“长裤”“10”：“裙子”，“11”：“面罩”，“12”：“标志”“13”：“条纹”}
    MALE = 0
    LONGHAIR = 1
    SUNGLASS = 2
    HAT = 3
    TSHIRT = 4
    LONGSLEEVE = 5
    FORMAL = 6
    SHORTS = 7
    JEANS = 8
    LONGPANTS = 9
    SKIRT = 10
    FACEMASK = 11
    LOGO = 12
    STRIPE = 13

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in WiderAttributes]

    @staticmethod
    def list_attributes(opt):
        out_rec = opt.specified_recognizable_attrs

        def fuc(ar):
            if str(ar) in out_rec:
                return Attribute(ar, AttributeType.BINARY, 1, rec_trainable=True)
            else:
                return Attribute(ar, AttributeType.BINARY, 1, rec_trainable=False)

        attrs_spc = filter(lambda x: str(x) in opt.specified_attrs,
                           [attr for attr in WiderAttributes])
        return list(map(fuc, attrs_spc))


class BerkeleyAttributes(Enum):
    # { 0”：“男性”，“1”：“长发”，“2”：“太阳镜”“3”：“帽子”，“4”：“T-shirt”，“5”：“长袖”,
    # “6”：“短裤”，“7”：“牛仔裤”“8”：“长裤”}
    MALE = 0
    LONGHAIR = 1
    SUNGLASS = 2
    HAT = 3
    TSHIRT = 4
    LONGSLEEVE = 5
    SHORTS = 6
    JEANS = 7
    LONGPANTS = 8

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in BerkeleyAttributes]

    @staticmethod
    def list_attributes(opt):
        out_rec = opt.specified_recognizable_attrs

        def fuc(ar):
            if str(ar) in out_rec:
                return Attribute(ar, AttributeType.BINARY, 1, rec_trainable=True)
            else:
                return Attribute(ar, AttributeType.BINARY, 1, rec_trainable=False)

        attrs_spc = filter(lambda x: str(x) in opt.specified_attrs,
                           [attr for attr in BerkeleyAttributes])
        return list(map(fuc, attrs_spc))


class ErisedAttributes(Enum):
    GENDER = 0
    AGE = 1
    AGE_GROUP = 2
    # DRESS = 0
    # GLASSES = 1
    # UNDERCUT = 2
    # GREASY = 3
    # PREGNANT = 4
    # AGE = 5
    # FIGURE = 6
    # HAIRCOLOR = 7
    # ALOPECIA = 8
    # TOTTOO = 9
    # CARRY_KIDS = 10
    # GENDER = 11

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in ErisedAttributes]


class NewAttributes(Enum):
    # 是否紧身(衣服) + 衣服种类 + 是否高跟 + 鞋子种类 + 是否紧身(裤子) + 裤子种类 + 是否有帽子 +
    # 是否有围巾 + 头发是否遮挡脖子 + 高领是否遮挡脖子 + 是否高发髻 + 是否大肚腩
    yifujinshen_yesno = 0
    yifu_zhonglei = 1
    gaogen_yesno = 2
    xiezi_zhonglei = 3
    kuzijinshen_yesno = 4
    kuzi_zhonglei = 5
    maozi_yesno = 6
    weijin_yesno = 7
    toufadangbozi_yesno = 8
    gaolingdangbozi_yesno = 9
    gaofaji_yesno = 10
    daduzi_yesno = 11

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def num_of_class(at):
        if at == 'yifu_zhonglei':
            return 23
            # return 10
        elif at == 'xiezi_zhonglei':
            return 29
            # return 4
        elif at == 'kuzi_zhonglei':
            return 12
            # return 10
        else:
            return 1

    @staticmethod
    def names():
        return [a.name.lower() for a in NewAttributes]

    @staticmethod
    def list_attributes(opt):
        out_rec = opt.specified_recognizable_attrs
        size = 14 if opt.conv.startswith('vgg') else 7
        cols = np.arange(0, size, 1, np.float)[np.newaxis, :] + 0.5
        rows = np.arange(0, size, 1, np.float)[:, np.newaxis] + 0.5
        if opt.at_level == 'wide':
            # 1
            yifujinshen_center = [[1 / 2 * size, 1 / 3 * size], [1 / 6 * size, 1 / 2 * size],
                                  [5 / 6 * size, 1 / 2 * size]]
            yifujinshen_sigma = [[2 * size / 7, 3 * size / 7], [2 * size / 7, 4 * size / 7],
                                 [2 * size / 7, 4 * size / 7]]
            yifujinshen_middle = np.exp(
                - (np.abs((cols - yifujinshen_center[0][0])) ** 3 / (yifujinshen_sigma[0][0] ** 2) +
                   np.abs((rows - yifujinshen_center[0][1])) ** 3 / (
                           (yifujinshen_sigma[0][1] * 2) ** 2)))
            yifujinshen_left = np.exp(
                - (np.abs((cols - yifujinshen_center[1][0])) ** 3 / (yifujinshen_sigma[1][0] ** 2) +
                   np.abs((rows - yifujinshen_center[1][1])) ** 3 / (yifujinshen_sigma[1][1] ** 2)))
            yifujinshen_right = np.exp(
                - (np.abs((cols - yifujinshen_center[2][0])) ** 3 / (yifujinshen_sigma[2][0] ** 2) +
                   np.abs((rows - yifujinshen_center[2][1])) ** 3 / (yifujinshen_sigma[2][1] ** 2)))
            yifujinshen_sum = yifujinshen_middle + yifujinshen_left + yifujinshen_right
            yifujinshen = (yifujinshen_sum / np.max(yifujinshen_sum)).reshape(-1)

            # 2
            kuzijinshen_center = [1 / 2 * size, 2 / 3 * size]
            kuzijinshen_sigma = [15 * size / 7, 15 * size / 7]
            kuzijinshen = np.exp(- (np.abs((cols - kuzijinshen_center[0])) ** 4 / (kuzijinshen_sigma[0] ** 2) +
                                    np.abs((rows - kuzijinshen_center[1])) ** 4 / (kuzijinshen_sigma[1] ** 2)))
            kuzijinshen = (kuzijinshen / np.max(kuzijinshen)).reshape(-1)

            # 3
            maozi_center = [[1 / 2 * size, 0 * size], [1 / 2 * size, 1 / 3 * size]]
            maozi_sigma = [5 * size / 7, 1.5 * size / 7]
            maozi_up = np.exp(- ((np.abs(cols - maozi_center[0][0])) ** 3 / (maozi_sigma[0] ** 2) +
                                 (rows - maozi_center[0][1]) ** 2 / (maozi_sigma[1] ** 2)))

            maozi_down = np.exp(- ((np.abs(cols - maozi_center[1][0])) ** 3 / (maozi_sigma[0] ** 2) +
                                   (rows - maozi_center[1][1]) ** 2 / (maozi_sigma[1] ** 2)))
            maozi = maozi_up + maozi_down
            maozi = (maozi / np.max(maozi)).reshape(-1)

            # 4
            gaolingdangbozi_center = [1 / 2 * size, 1 / 6 * size]
            gaolingdangbozi_sigma = [5 * size / 7, 2 * size / 7]
            gaolingdangbozi = np.exp(
                - ((np.abs(cols - gaolingdangbozi_center[0])) ** 3 / (gaolingdangbozi_sigma[0] ** 2) +
                   (rows - gaolingdangbozi_center[1]) ** 2 / (gaolingdangbozi_sigma[1] ** 2)))
            gaolingdangbozi = (gaolingdangbozi / np.max(gaolingdangbozi)).reshape(-1)

            # 5
            gaofaji_center = [1 / 2 * size, 1 / 24 * size]
            gaofaji_sigma = [5 * size / 7, 2 * size / 7]
            gaofaji = np.exp(- ((np.abs(cols - gaofaji_center[0])) ** 3 / (gaofaji_sigma[0] ** 2) +
                                (rows - gaofaji_center[1]) ** 2 / (gaofaji_sigma[1] ** 2)))
            gaofaji = (gaofaji / np.max(gaofaji)).reshape(-1)
        else:
            # 1
            yifujinshen_center = [[1 / 2 * size, 1 / 3 * size], [1 / 6 * size, 1 / 2 * size],
                                  [5 / 6 * size, 1 / 2 * size]]
            yifujinshen_sigma = [[1.5 * size / 7, 2.5 * size / 7], [1.5 * size / 7, 3.5 * size / 7],
                                 [1.5 * size / 7, 3.5 * size / 7]]
            yifujinshen_middle = np.exp(
                - (np.abs((cols - yifujinshen_center[0][0])) ** 3 / (yifujinshen_sigma[0][0] ** 2) +
                   np.abs((rows - yifujinshen_center[0][1])) ** 3 / (
                           (yifujinshen_sigma[0][1] * 2) ** 2)))
            yifujinshen_left = np.exp(
                - (np.abs((cols - yifujinshen_center[1][0])) ** 3 / (yifujinshen_sigma[1][0] ** 2) +
                   np.abs((rows - yifujinshen_center[1][1])) ** 3 / (yifujinshen_sigma[1][1] ** 2)))
            yifujinshen_right = np.exp(
                - (np.abs((cols - yifujinshen_center[2][0])) ** 3 / (yifujinshen_sigma[2][0] ** 2) +
                   np.abs((rows - yifujinshen_center[2][1])) ** 3 / (yifujinshen_sigma[2][1] ** 2)))
            yifujinshen_sum = yifujinshen_middle + yifujinshen_left + yifujinshen_right
            yifujinshen = (yifujinshen_sum / np.max(yifujinshen_sum)).reshape(-1)

            # 2
            kuzijinshen_center = [1 / 2 * size, 2 / 3 * size]
            kuzijinshen_sigma = [10 * size / 7, 10 * size / 7]
            kuzijinshen = np.exp(- (np.abs((cols - kuzijinshen_center[0])) ** 4 / (kuzijinshen_sigma[0] ** 2) +
                                    np.abs((rows - kuzijinshen_center[1])) ** 4 / (kuzijinshen_sigma[1] ** 2)))
            kuzijinshen = (kuzijinshen / np.max(kuzijinshen)).reshape(-1)

            # 3
            maozi_center = [[1 / 2 * size, 0 * size], [1 / 2 * size, 1 / 4 * size]]
            maozi_sigma = [3 * size / 7, 1 * size / 7]
            maozi_up = np.exp(- ((np.abs(cols - maozi_center[0][0])) ** 3 / (maozi_sigma[0] ** 2) +
                                 (rows - maozi_center[0][1]) ** 2 / (maozi_sigma[1] ** 2)))

            maozi_down = np.exp(- ((np.abs(cols - maozi_center[1][0])) ** 3 / (maozi_sigma[0] ** 2) +
                                   (rows - maozi_center[1][1]) ** 2 / (maozi_sigma[1] ** 2)))
            maozi = maozi_up + maozi_down
            maozi = (maozi / np.max(maozi)).reshape(-1)

            # 4
            gaolingdangbozi_center = [1 / 2 * size, 1 / 6 * size]
            gaolingdangbozi_sigma = [3 * size / 7, 1 * size / 7]
            gaolingdangbozi = np.exp(
                - ((np.abs(cols - gaolingdangbozi_center[0])) ** 3 / (gaolingdangbozi_sigma[0] ** 2) +
                   (rows - gaolingdangbozi_center[1]) ** 2 / (gaolingdangbozi_sigma[1] ** 2)))
            gaolingdangbozi = (gaolingdangbozi / np.max(gaolingdangbozi)).reshape(-1)

            # 5
            gaofaji_center = [1 / 2 * size, 1 / 24 * size]
            gaofaji_sigma = [3 * size / 7, 1 * size / 7]
            gaofaji = np.exp(- ((np.abs(cols - gaofaji_center[0])) ** 3 / (gaofaji_sigma[0] ** 2) +
                                (rows - gaofaji_center[1]) ** 2 / (gaofaji_sigma[1] ** 2)))
            gaofaji = (gaofaji / np.max(gaofaji)).reshape(-1)


        hot_map = dict()
        hot_map[NewAttributes.yifujinshen_yesno] = yifujinshen
        hot_map[NewAttributes.kuzijinshen_yesno] = kuzijinshen
        hot_map[NewAttributes.maozi_yesno] = maozi
        hot_map[NewAttributes.gaolingdangbozi_yesno] = gaolingdangbozi
        hot_map[NewAttributes.gaofaji_yesno] = gaofaji

        def fuc(ar):
            at = hot_map[ar] if ar in hot_map else []
            if str(ar) in out_rec:
                return Attribute(ar, AttributeType.BINARY if str(ar).endswith('yesno') else AttributeType.MULTICLASS,
                                 NewAttributes.num_of_class(str(ar)), at, rec_trainable=True)
            else:
                return Attribute(ar, AttributeType.BINARY if str(ar).endswith('yesno') else AttributeType.MULTICLASS,
                                 NewAttributes.num_of_class(str(ar)), at, rec_trainable=False)

        attrs_spc = filter(lambda x: str(x) in opt.specified_attrs,
                           [attr for attr in NewAttributes])
        return list(map(fuc, attrs_spc))


class AttributeType(Enum):
    BINARY = 0
    MULTICLASS = 1
    NUMERICAL = 2


class Attribute:
    def __init__(self, key, tp, bn, at, rec_trainable=False):
        assert isinstance(key, Enum)
        assert isinstance(tp, AttributeType)
        self.key = key
        self.name = str(key)
        self.data_type = tp
        self.rec_trainable = rec_trainable
        self.branch_num = bn
        self.at = at
        self.at_coe = np.sum(at)

    def __str__(self):
        return self.name




