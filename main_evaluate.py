import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='screw')
args = parser.parse_args()


def do_evaluate_encoder_multiK(obj):
    from codes.inspection import eval_encoder_NN_multiK
    from codes.vit import ViT

    enc_16 = ViT(
        image_size=48,
        patch_size=16,
        channels=3,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda(1)

    enc_32 = ViT(
        image_size=96,
        patch_size=32,
        channels=3,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda(1)
    enc_16.load(obj, 16)
    enc_16.eval()
    enc_32.load(obj, 32)
    enc_32.eval()
    results = eval_encoder_NN_multiK(enc_16, enc_32, obj)

    det_64 = results['det_64'] * 100
    seg_64 = results['seg_64'] * 100

    det_32 = results['det_32'] * 100
    seg_32 = results['seg_32'] * 100

    det_sum = results['det_sum'] * 100
    seg_sum = results['seg_sum'] * 100

    det_mult = results['det_mult'] * 100
    seg_mult = results['seg_mult'] * 100

    print(
        f'| K64 | Det: {det_64:.3f} Seg:{seg_64:.3f}'
        f'| K32 | Det: {det_32:.3f} Seg:{seg_32:.3f} '
        f'| sum | Det: {det_sum:.3f} Seg:{seg_sum:.3f} '
        f'| mult | Det: {det_mult:.3f} Seg:{seg_mult:.3f} ({obj})')


#########################


def main():
    print('test start')
    do_evaluate_encoder_multiK(args.obj)
    print('test end')


if __name__ == '__main__':
    main()
