from vecssl.data.svg_tensor import SVGTensor


class _DefaultConfig:
    """
    Model config.
    """

    def __init__(self):
        self.args_dim = 256  # Coordinate numericalization, default: 256 (8-bit)
        self.n_args = (
            11  # Tensor nb of arguments, default: 11 (rx,ry,phi,fA,fS,qx1,qy1,qx2,qy2,x1,x2)
        )
        self.n_commands = len(SVGTensor.COMMANDS_SIMPLIFIED)  # m, l, c, a, EOS, SOS, z

        self.dropout = 0.1  # Dropout rate used in basic layers and Transformers

        self.model_type = "transformer"  # "transformer" ("lstm" implementation is work in progress)

        self.encode_stages = 2  # One-stage or two-stage: 1 | 2
        self.decode_stages = 1  # One-stage or two-stage: 1 | 2

        self.use_resnet = True  # Use extra fully-connected residual blocks after Encoder

        self.use_vae = (
            True  # Sample latent vector (with reparametrization trick) or use encodings directly
        )

        self.pred_mode = (
            "one_shot"  # Feed-forward (one-shot) or autogressive: "one_shot" | "autoregressive"
        )
        self.rel_targets = False  # Predict coordinates in relative or absolute format

        self.label_condition = False  # Make all blocks conditional on the label
        self.n_labels = 100  # Number of labels (when used)
        self.dim_label = 64  # Label embedding dimensionality

        self.self_match = False  # Use Hungarian (self-match) or Ordered assignment

        self.n_layers = 4  # Number of Encoder blocks
        self.n_layers_decode = 4  # Number of Decoder blocks
        self.n_heads = 8  # Transformer config: number of heads
        self.dim_feedforward = 512  # Transformer config: FF dimensionality
        self.d_model = 256  # Transformer config: model dimensionality

        self.dim_z = 256  # Latent vector dimensionality

        self.max_num_groups = 8  # Number of paths (N_P)
        self.max_seq_len = 30  # Number of commands (N_C)
        self.max_total_len = (
            self.max_num_groups * self.max_seq_len
        )  # Concatenated sequence length for baselines

        self.num_groups_proposal = self.max_num_groups  # Number of predicted paths, default: N_P

        self.lr = 1e-4
        self.batch_size = 64
        self.epochs = 100

        self.DINO_layer = -1  # DINO layer to use (-1 is the last layer)

    def get_model_args(self):
        model_args = []

        model_args += (
            ["commands_grouped", "args_grouped"]
            if self.encode_stages <= 1
            else ["commands", "args"]
        )

        if self.rel_targets:
            model_args += (
                ["commands_grouped", "args_rel_grouped"]
                if self.decode_stages == 1
                else ["commands", "args_rel"]
            )
        else:
            model_args += (
                ["commands_grouped", "args_grouped"]
                if self.decode_stages == 1
                else ["commands", "args"]
            )

        if self.label_condition:
            model_args.append("label")

        return model_args


class SketchRNN(_DefaultConfig):
    # LSTM - Autoregressive - One-stage
    def __init__(self):
        super().__init__()

        self.model_type = "lstm"

        self.pred_mode = "autoregressive"
        self.rel_targets = True


class Sketchformer(_DefaultConfig):
    # Transformer - Autoregressive - One-stage
    def __init__(self):
        super().__init__()

        self.pred_mode = "autoregressive"
        self.rel_targets = True


class OneStageOneShot(_DefaultConfig):
    # Transformer - One-shot - One-stage
    def __init__(self):
        super().__init__()

        self.encode_stages = 1
        self.decode_stages = 1


class Hierarchical(_DefaultConfig):
    # Transformer - One-shot - Two-stage - Ordered
    def __init__(self):
        super().__init__()

        self.encode_stages = 2
        self.decode_stages = 2


class HierarchicalSelfMatching(_DefaultConfig):
    # Transformer - One-shot - Two-stage - Hungarian
    def __init__(self):
        super().__init__()

        self.encode_stages = 2
        self.decode_stages = 2
        self.self_match = True


class ContrastiveConfig(_DefaultConfig):
    # Contrastive model
    def __init__(self):
        super().__init__()
        self.contrastive_logit_scale = 0.07
        self.d_joint = 768
        self.use_group = True
        self.d_model = self.d_joint
        self.use_precomputed_dino = True
        self.dino_dir = "/oscar/scratch/zzhan215/google_fonts_processed_reduced/dino"


class JepaConfig(_DefaultConfig):
    def __init__(self):
        super().__init__()

        self.encode_stages = 2

        self.d_joint = 768
        # self.d_model = self.d_joint

        self.use_resnet = False
        self.predictor_type = "mlp"

        self.predictor_transformer_num_heads = 4
        self.predictor_transformer_num_layers = 2
        self.predictor_transformer_hidden_dim = 128
        self.predictor_transformer_dropout = 0.3

        self.predictor_mlp_num_layers = 2
        self.predictor_mlp_hidden_dim = 768
        self.predictor_mlp_dropout = 0.1

        # Precomputed DINO embeddings support
        self.use_precomputed_dino = False


class SVGMAEConfig(_DefaultConfig):
    """Config for SVG-only Masked Autoencoder"""

    def __init__(self):
        super().__init__()

        self.encode_stages = 2  # Use 2-stage structure for per-group encoding
        self.decode_stages = 1  # Single-stage decoder for reconstructing tokens
        self.use_vae = False  # No VAE for MAE
        self.use_resnet = False

        # Masking ratio
        self.mask_ratio_svg = 0.75  # Mask 75% of SVG groups

        # MAE encoder (processes visible groups + CLS)
        self.mae_depth = 4
        self.mae_num_heads = 8
        self.mae_mlp_ratio = 4.0
        self.mae_dropout = 0.1

        # Loss weights (using SVGLoss pattern)
        self.loss_cmd_weight = 1.0
        self.loss_args_weight = 2.0


class MultiMAEConfig(_DefaultConfig):
    """Config for SVG + Image Multi-modal Masked Autoencoder"""

    def __init__(self):
        super().__init__()

        self.encode_stages = 2  # Use 2-stage structure for per-group encoding
        self.decode_stages = 1  # Single-stage decoder for reconstructing tokens
        self.use_vae = False  # No VAE for MAE
        self.use_resnet = False

        # Masking ratios
        self.mask_ratio_svg = 0.5  # Mask 50% of SVG groups
        self.mask_ratio_img = 0.75  # Mask 75% of image patches

        # Shared MAE encoder (processes visible SVG + visible image + CLS)
        self.mae_depth = 8
        self.mae_num_heads = 8
        self.mae_mlp_ratio = 4.0
        self.mae_dropout = 0.1

        # DINO configuration
        self.dino_model_name = "facebook/dinov2-base"
        self.train_dino = False
        self.use_precomputed_dino_patches = False

        # Image projection (768 -> d_model)
        self.img_proj_dim = 768  # DINO hidden size

        # Loss weights
        self.loss_cmd_weight = 1.0
        self.loss_args_weight = 2.0
