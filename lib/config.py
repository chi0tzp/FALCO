########################################################################################################################
## Basic configuration file.                                                                                          ##
##                                                                                                                    ##
##                                                                                                                    ##
########################################################################################################################

########################################################################################################################
##                                                                                                                    ##
##                                                    [ Datasets ]                                                    ##
##                                                                                                                    ##
########################################################################################################################
DATASETS = {
    'celeba': 'datasets/CelebA/',
    'celebahq': 'datasets/CelebA-HQ/',
    'lfw': 'datasets/LFW/',
}

CelebA_classes = ('5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                  'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                  'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                  'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                  'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                  'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young')


########################################################################################################################
##                                                                                                                    ##
##                                                      [ FaRL ]                                                      ##
##                                                                                                                    ##
########################################################################################################################
# Choose pre-trained FaRL model (epoch 16 or 64)
FARL_EP = 64
FARL_PRETRAIN_MODEL = 'FaRL-Base-Patch16-LAIONFace20M-ep{}.pth'.format(FARL_EP)

FARL = ('https://www.dropbox.com/s/xxhmvo3q7avlcac/farl.tar?dl=1',
        '1d67cc6fd3cea9fdd7ec6af812a32e6b02374162d02137dd80827283d496b2d8')

########################################################################################################################
##                                                                                                                    ##
##                                                    [ ArcFace ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
ARCFACE = ('https://www.dropbox.com/s/idulblr8pdrmbq1/arcface.tar?dl=1',
           'edd5854cacd86c17a78a11f70ab8c49bceffefb90ee070754288fa7ceadcdfb2')

########################################################################################################################
##                                                                                                                    ##
##                                                     [ E4E ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
E4E = ('https://www.dropbox.com/s/1jujsdr6ytzilym/e4e.tar?dl=1',
       'b4a95155f2bebbb229b7dfc914fe937753b9a5b8de9a837875f9fbcacf8bb287')

########################################################################################################################
##                                                                                                                    ##
##                                                     [ SFD ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
SFD = ('https://www.dropbox.com/scl/fi/eo6c8prmvuhpvpx7sh4u8/sfd.tar?rlkey=7bpo0kxennilgz2kglwpgozy5&dl=1',
       '2bea5f1c10110e356eef3f4efd45169100b9c7704eb6e6abd309df58f34452d4')

########################################################################################################################
##                                                                                                                    ##
##                                             [ GenForce GAN Generators ]                                            ##
##                                                                                                                    ##
########################################################################################################################
GENFORCE = ('https://www.dropbox.com/scl/fi/yec5wgg8j388fc0saigbj/genforce.tar?rlkey=kkhkkxvnc985746ichtdyct3f&dl=1',
            '63284b4f4ffeac38037061fd175c462afff82bbe570ed80092720a724a67a6dc')

GENFORCE_MODELS = {
    'stylegan2_ffhq1024': ('stylegan2_ffhq1024.pth', 1024),
    'stylegan2_ffhq512': ('stylegan2_ffhq512.pth', 512)
}

STYLEGAN_LAYERS = {
    'stylegan2_ffhq1024': 18,
    'stylegan2_ffhq512': 16
}

GAN_BASE_LATENT_DIM = {
    'stylegan2_ffhq1024': 512,
    'stylegan2_ffhq512': 512
}


STYLEGAN2_STYLE_SPACE_LAYERS = {
    'stylegan2_ffhq1024':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            'style08': 512,  # 'layer8'  : '64x64/Conv1'
            'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            'style10': 256,  # 'layer10' : '128x128/Conv1'
            'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            'style12': 128,  # 'layer12' : '256x256/Conv1'
            'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            'style14': 64,   # 'layer14' : '512x512/Conv1'
            'style15': 64,   # 'layer15' : '1024x1024/Conv0_up'
            'style16': 32    # 'layer16' : '1024x1024/Conv1'
        },
    'stylegan2_ffhq512':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            'style08': 512,  # 'layer8'  : '64x64/Conv1'
            'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            'style10': 256,  # 'layer10' : '128x128/Conv1'
            'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            'style12': 128,  # 'layer12' : '256x256/Conv1'
            'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            'style14': 64,   # 'layer14' : '512x512/Conv1'
        }
}


STYLEGAN2_STYLE_SPACE_TARGET_LAYERS = {
    'stylegan2_ffhq1024':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            # 'style08': 512,  # 'layer8'  : '64x64/Conv1'
            # 'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            # 'style10': 256,  # 'layer10' : '128x128/Conv1'
            # 'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            # 'style12': 128,  # 'layer12' : '256x256/Conv1'
            # 'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            # 'style14': 64,   # 'layer14' : '512x512/Conv1'
            # 'style15': 64,   # 'layer15' : '1024x1024/Conv0_up'
            # 'style16': 32    # 'layer16' : '1024x1024/Conv1'
        },
    'stylegan2_ffhq512':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            # 'style08': 512,  # 'layer8'  : '64x64/Conv1'
            # 'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            # 'style10': 256,  # 'layer10' : '128x128/Conv1'
            # 'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            # 'style12': 128,  # 'layer12' : '256x256/Conv1'
            # 'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            # 'style14': 64,   # 'layer14' : '512x512/Conv1'
        }
}
