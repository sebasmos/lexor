import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from monai.networks.nets import ViT, ResNet, EfficientNetBN
import time
from typing import Optional, Dict, Any

# =============================================================================
# FAST ENCODER IMPLEMENTATIONS
# =============================================================================

class FastViTEncoder(nn.Module):
    """Lightweight ViT with reduced complexity for fast zoom-out inference"""
    
    def __init__(self, 
                 img_size=(32, 256, 256),
                 patch_size=(8, 32, 32),
                 hidden_size=384,
                 num_layers=6,
                 num_heads=6,
                 output_dim=768):
        super().__init__()
        
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=hidden_size * 4,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed='perceptron',
            classification=False,
            dropout_rate=0.0,
        )
        
        # Project to expected output dimensions
        self.projection = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        # Input: [B, 1, 32, 256, 256]
        features, _ = self.vit(x)  # [B, hidden_size, feat_d, feat_h, feat_w]
        
        # Project features to match expected dimensions
        B, C, D, H, W = features.shape
        features = features.permute(0, 2, 3, 4, 1)  # [B, D, H, W, hidden_size]
        features = self.projection(features)         # [B, D, H, W, output_dim]
        features = features.permute(0, 4, 1, 2, 3)  # [B, output_dim, D, H, W]
        
        return features


class UltraFast2_5D(nn.Module):
    """Ultra-fast 2.5D processing using lightweight 2D models"""
    
    def __init__(self, 
                 model_name='mobilenetv3_small_100',
                 num_slices=3,
                 output_dim=768):
        super().__init__()
        self.num_slices = num_slices
        self.output_dim = output_dim
        
        # Use fastest TIMM model
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=num_slices,
            num_classes=output_dim,
            global_pool='avg'
        )
        
    def forward(self, x):
        # Input: [B, 1, D, H, W]
        B, C, D, H, W = x.shape
        
        # Strategic slice selection for speed
        if self.num_slices == 3:
            # Take beginning, middle, end slices
            slice_indices = [D//4, D//2, 3*D//4]
        elif self.num_slices == 5:
            # More slices for better representation
            slice_indices = [D//6, D//3, D//2, 2*D//3, 5*D//6]
        else:
            # Evenly spaced slices
            slice_indices = torch.linspace(0, D-1, self.num_slices).long().tolist()
        
        # Extract and stack slices
        slices = []
        for idx in slice_indices:
            slices.append(x[:, :, idx, :, :])  # [B, 1, H, W]
        
        multi_slice = torch.cat(slices, dim=1)  # [B, num_slices, H, W]
        
        # Process as 2D image with multiple channels
        features = self.backbone(multi_slice)  # [B, output_dim]
        
        # Reshape to match 3D encoder output format
        # Need to create spatial dimensions for compatibility
        feat_size = max(1, min(8, D//4))  # Adaptive feature size
        features = features.view(B, self.output_dim, 1, 1, 1)
        features = features.expand(B, self.output_dim, feat_size, feat_size, feat_size)
        
        return features


class MobileNet3D(nn.Module):
    """3D adaptation of MobileNet for fast inference"""
    
    def __init__(self, output_dim=768, width_mult=1.0):
        super().__init__()
        
        # Define MobileNet-like architecture for 3D
        def conv_bn_relu(inp, oup, kernel_size, stride, padding=0):
            return nn.Sequential(
                nn.Conv3d(inp, oup, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU6(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # Depthwise convolution
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                nn.ReLU6(inplace=True),
                
                # Pointwise convolution
                nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU6(inplace=True),
            )
        
        # Calculate channel sizes
        def make_divisible(x, divisible_by=8):
            return int(max(divisible_by, int(x + divisible_by / 2) // divisible_by * divisible_by))
        
        input_channel = make_divisible(32 * width_mult)
        last_channel = make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        
        # Build network
        self.features = nn.Sequential(
            conv_bn_relu(1, input_channel, 3, 2, 1),  # First conv
            
            # MobileNet blocks
            conv_dw(input_channel, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            
            # Reduce to output dimensions
            conv_bn_relu(512, last_channel, 1, 1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(last_channel, output_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # Reshape to spatial format for compatibility
        return x.view(x.size(0), -1, 1, 1, 1)


class HybridCNNViT(nn.Module):
    """Hybrid approach: CNN stem + lightweight transformer"""
    
    def __init__(self, output_dim=768):
        super().__init__()
        
        # CNN stem to reduce spatial dimensions quickly
        self.cnn_stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # [64, 8, 64, 64]
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),  # [128, 4, 32, 32]
        )
        
        # Lightweight transformer on reduced resolution
        self.transformer = ViT(
            in_channels=128,
            img_size=(4, 32, 32),
            patch_size=(2, 8, 8),
            hidden_size=384,
            mlp_dim=1536,
            num_layers=4,
            num_heads=6,
            pos_embed='perceptron',
            classification=False,
        )
        
        # Output projection
        self.output_proj = nn.Conv3d(384, output_dim, kernel_size=1)
        
    def forward(self, x):
        # Fast CNN preprocessing
        x = self.cnn_stem(x)
        
        # Lightweight attention
        x, _ = self.transformer(x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x


class FastResNet3D(nn.Module):
    """Minimal ResNet3D for speed"""
    
    def __init__(self, output_dim=768):
        super().__init__()
        
        # Minimal ResNet architecture
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Only 2 ResNet layers for speed
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, output_dim)
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(self._make_block(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(planes, planes))
        return nn.Sequential(*layers)
    
    def _make_block(self, inplanes, planes, stride=1):
        return nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(planes),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x.view(x.size(0), -1, 1, 1, 1)


# =============================================================================
# ENCODER FACTORY
# =============================================================================

class FastEncoderFactory:
    """Factory to create different fast encoders"""
    
    ENCODER_CONFIGS = {
        'fast_vit': {
            'class': FastViTEncoder,
            'params': {'hidden_size': 384, 'num_layers': 6, 'num_heads': 6},
            'description': 'Lightweight ViT with reduced complexity'
        },
        'ultra_fast_vit': {
            'class': FastViTEncoder,
            'params': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4, 'patch_size': (16, 64, 64)},
            'description': 'Ultra-lightweight ViT for maximum speed'
        },
        'mobilenet_2_5d': {
            'class': UltraFast2_5D,
            'params': {'model_name': 'mobilenetv3_small_100', 'num_slices': 3},
            'description': '2.5D processing with MobileNet'
        },
        'efficientnet_2_5d': {
            'class': UltraFast2_5D,
            'params': {'model_name': 'efficientnet_lite0', 'num_slices': 5},
            'description': '2.5D processing with EfficientNet'
        },
        'mobilenet_3d': {
            'class': MobileNet3D,
            'params': {'width_mult': 0.5},
            'description': '3D MobileNet for mobile-optimized inference'
        },
        'hybrid_cnn_vit': {
            'class': HybridCNNViT,
            'params': {},
            'description': 'CNN stem + lightweight transformer'
        },
        'fast_resnet3d': {
            'class': FastResNet3D,
            'params': {},
            'description': 'Minimal ResNet3D for speed'
        }
    }
    
    @classmethod
    def create_encoder(cls, encoder_type: str, **kwargs) -> nn.Module:
        """Create encoder by type"""
        if encoder_type not in cls.ENCODER_CONFIGS:
            available = list(cls.ENCODER_CONFIGS.keys())
            raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {available}")
        
        config = cls.ENCODER_CONFIGS[encoder_type]
        params = {**config['params'], **kwargs}
        
        return config['class'](**params)
    
    @classmethod
    def list_encoders(cls) -> Dict[str, str]:
        """List available encoders with descriptions"""
        return {name: config['description'] for name, config in cls.ENCODER_CONFIGS.items()}


# =============================================================================
# MODIFIED SEGVOL INTEGRATION
# =============================================================================

def _build_sam_with_fast_encoder(
    image_encoder_type='vit',
    embed_dim=768,
    patch_size=(4, 16, 16),
    checkpoint=None,
    image_size=(32, 256, 256),
    fast_encoder_type='fast_vit',  # New parameter
    **encoder_kwargs
):
    """Modified _build_sam function with fast encoder support"""
    
    if fast_encoder_type != 'original':
        # Use fast encoder
        image_encoder = FastEncoderFactory.create_encoder(
            fast_encoder_type, 
            output_dim=embed_dim,
            **encoder_kwargs
        )
        print(f"Using fast encoder: {fast_encoder_type}")
    else:
        # Original ViT encoder
        mlp_dim = 3072
        num_layers = 12
        num_heads = 12
        pos_embed = 'perceptron'
        dropout_rate = 0.0
        
        image_encoder = ViT(
            in_channels=1,
            img_size=image_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=False,
            dropout_rate=dropout_rate,
        )
        print("Using original ViT encoder")
    
    # Calculate image embedding size (needed for other components)
    if fast_encoder_type in ['mobilenet_2_5d', 'efficientnet_2_5d', 'mobilenet_3d', 'fast_resnet3d']:
        # These encoders output single spatial dimension
        image_embedding_size = [1, 1, 1]
    else:
        # Calculate based on patch size
        import numpy as np
        image_embedding_size = [int(item) for item in (np.array(image_size) / np.array(patch_size))]
    
    # Load checkpoint if provided
    if checkpoint is not None and fast_encoder_type == 'original':
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')['state_dict']
            encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
        image_encoder.load_state_dict(encoder_dict)
        print(f'Loaded encoder checkpoint: {checkpoint}')
    
    # Import other SAM components (assuming they exist in your codebase)
    from your_sam_module import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer
    
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=image_size,
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            image_encoder_type=image_encoder_type,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            image_size=np.array(image_size),
            patch_size=np.array(patch_size),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    
    sam.eval()
    return sam


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

class EncoderBenchmark:
    """Utility to benchmark different encoders"""
    
    def __init__(self, input_shape=(1, 1, 32, 256, 256), device='cuda'):
        self.input_shape = input_shape
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dummy_input = torch.randn(input_shape).to(self.device)
    
    def benchmark_encoder(self, encoder, num_runs=100, warmup_runs=10):
        """Benchmark single encoder"""
        encoder = encoder.to(self.device)
        encoder.eval()
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = encoder(self.dummy_input)
        
        # Benchmark
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = encoder(self.dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Memory usage
        if self.device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_used = None
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': 1.0 / avg_time,
            'memory_mb': memory_used,
            'output_shape': tuple(output.shape)
        }
    
    def compare_encoders(self, encoder_types=None):
        """Compare multiple encoders"""
        if encoder_types is None:
            encoder_types = list(FastEncoderFactory.ENCODER_CONFIGS.keys())
        
        results = {}
        
        print("Benchmarking encoders...")
        print("-" * 80)
        
        for encoder_type in encoder_types:
            try:
                encoder = FastEncoderFactory.create_encoder(encoder_type)
                result = self.benchmark_encoder(encoder)
                results[encoder_type] = result
                
                desc = FastEncoderFactory.ENCODER_CONFIGS[encoder_type]['description']
                print(f"{encoder_type:20s} | {result['avg_time_ms']:8.2f} ms | "
                      f"{result['fps']:6.1f} FPS | {desc}")
                
            except Exception as e:
                print(f"{encoder_type:20s} | ERROR: {str(e)}")
                results[encoder_type] = {'error': str(e)}
        
        return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of fast encoders"""
    
    # List available encoders
    print("Available fast encoders:")
    for name, desc in FastEncoderFactory.list_encoders().items():
        print(f"  {name}: {desc}")
    print()
    
    # Create and test specific encoder
    encoder = FastEncoderFactory.create_encoder('fast_vit')
    print(f"Created encoder: {encoder.__class__.__name__}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 32, 256, 256)
    with torch.no_grad():
        output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print()
    
    # Benchmark comparison
    benchmark = EncoderBenchmark()
    results = benchmark.compare_encoders(['fast_vit', 'mobilenet_2_5d', 'ultra_fast_vit'])
    
    # Find fastest encoder
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1]['avg_time_ms'])
        print(f"\nFastest encoder: {fastest[0]} ({fastest[1]['avg_time_ms']:.2f} ms)")


if __name__ == "__main__":
    main()
