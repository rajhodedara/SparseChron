import torch
from sparsechron.models.classifier import StaticDynamicClassifier
from sparsechron.models.gaussians import GaussianModel

def test_classifier_update():
    num_gaussians = 5
    classifier = StaticDynamicClassifier(num_gaussians)
    
    # d_pos shape (5, 3)
    d_pos = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ])
    
    classifier.update(d_pos)
    expected_norms = torch.tensor([1.0, 2.0, 3.0, 3**0.5, 0.0])
    assert torch.allclose(classifier.cum_deformation, expected_norms)

def test_classifier_update_with_mask():
    num_gaussians = 5
    classifier = StaticDynamicClassifier(num_gaussians)
    
    d_pos = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0]
    ])
    mask = torch.tensor([True, False, True, False, False])
    
    classifier.update(d_pos, mask)
    expected_norms = torch.tensor([1.0, 0.0, 2.0, 0.0, 0.0])
    assert torch.allclose(classifier.cum_deformation, expected_norms)

def test_classifier_reclassify():
    num_gaussians = 3
    classifier = StaticDynamicClassifier(num_gaussians)
    classifier.cum_deformation = torch.tensor([0.5, 1.5, 2.5])
    
    initial_values = {
        "positions": torch.zeros((num_gaussians, 3)),
        "scales": torch.zeros((num_gaussians, 3)),
        "rotations": torch.zeros((num_gaussians, 4)),
        "opacities": torch.zeros((num_gaussians, 1)),
        "sh_coeffs": torch.zeros((num_gaussians, 16, 3)),
    }
    model = GaussianModel(initial_values)
    
    classifier.reclassify(model, threshold=1.0)
    
    # Gaussian 0 (< 1.0) should be static, others dynamic
    expected_dynamic = torch.tensor([False, True, True])
    assert torch.equal(model.is_dynamic, expected_dynamic)
    
    # Check buffer reset
    assert torch.equal(classifier.cum_deformation, torch.zeros(num_gaussians))

def test_classifier_resize():
    num_gaussians = 3
    classifier = StaticDynamicClassifier(num_gaussians)
    classifier.resize(5)
    
    assert classifier.cum_deformation.shape[0] == 5
    assert torch.equal(classifier.cum_deformation, torch.zeros(5))

def test_classifier_device_mismatch():
    if not torch.cuda.is_available():
        return
        
    num_gaussians = 2
    classifier = StaticDynamicClassifier(num_gaussians)
    
    d_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device='cuda')
    classifier.update(d_pos)
    
    assert classifier.cum_deformation.device.type == 'cuda'
