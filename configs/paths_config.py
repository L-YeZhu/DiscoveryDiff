DATASET_PATHS = {
	'CELEBA': '/n/fs/yz-diff/UnseenDiffusion/imgs/human/',
	'AFHQ': '/n/fs/yz-diff/UnseenDiffusion/imgs/dogs/',
	'BEDROOM': '/n/fs/yz-diff/UnseenDiffusion/imgs/bedroom/',
	'CHURCH': '/n/fs/yz-diff/UnseenDiffusion/imgs/church/',
    'GALAXY': "/n/fs/yz-diff/Discovery/astro_data/galaxy/original/",
    'RADIATION':"/n/fs/yz-diff/Discovery/astro_data/radiation/original_ref/",
    'CUSTOME':'syn_imgs/',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhqdog_p2.pt",
    'BEDROOM': "pretrained/bedroom.ckpt",
    'CHURCH': "pretrained/church_outdoor.ckpt",
    'CELEBA': "pretrained/celeba_hq.ckpt",
    'shape_predictor': "pretrained/shape_predictor_68_face_landmarks.dat.bz2",
}


HYBRID_MODEL_PATHS = [
	'./checkpoint/human_face/curly_hair_t401.pth',
	'./checkpoint/human_face/with_makeup_t401.pth',
]

HYBRID_CONFIG = \
	{ 300: [0.4, 0.6, 0],
	    0: [0.15, 0.15, 0.7]}