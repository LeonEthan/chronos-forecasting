#!/usr/bin/env python3
"""
Compatibility checker for PyTorch and transformers versions.
This script helps diagnose and resolve version compatibility issues.
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, List


def check_versions() -> dict:
    """Check installed versions of key packages."""
    versions = {}
    
    try:
        import torch
        versions['torch'] = torch.__version__
    except ImportError:
        versions['torch'] = 'Not installed'
    
    try:
        import transformers
        versions['transformers'] = transformers.__version__
    except ImportError:
        versions['transformers'] = 'Not installed'
    
    try:
        import safetensors
        versions['safetensors'] = safetensors.__version__
    except ImportError:
        versions['safetensors'] = 'Not installed'
    
    return versions


def check_torch_compatibility(torch_version: str) -> Tuple[bool, str]:
    """Check if PyTorch version is compatible with latest transformers."""
    if torch_version == 'Not installed':
        return False, "PyTorch is not installed"
    
    try:
        # Parse version
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 6):
            return True, f"PyTorch {torch_version} is compatible"
        else:
            return False, f"PyTorch {torch_version} may have compatibility issues. Recommended: 2.6+"
    except Exception as e:
        return False, f"Could not parse PyTorch version: {e}"


def check_checkpoint_format(checkpoint_path: str) -> dict:
    """Check the format of checkpoint files."""
    if not checkpoint_path:
        return {"error": "No checkpoint path provided"}
    
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        return {"error": f"Checkpoint path does not exist: {checkpoint_path}"}
    
    result = {
        "path": checkpoint_path,
        "config_json": (checkpoint_dir / "config.json").exists(),
        "pytorch_model_bin": (checkpoint_dir / "pytorch_model.bin").exists(),
        "safetensors_files": list(checkpoint_dir.glob("*.safetensors")),
        "has_safetensors": len(list(checkpoint_dir.glob("*.safetensors"))) > 0
    }
    
    return result


def suggest_solutions(versions: dict, torch_compatible: bool) -> List[str]:
    """Suggest solutions based on the current setup."""
    solutions = []
    
    if not torch_compatible:
        solutions.append(
            "🔧 Upgrade PyTorch to version 2.6 or higher:\n"
            "   conda install pytorch>=2.6 -c pytorch\n"
            "   # or\n"
            "   pip install torch>=2.6"
        )
    
    if versions['safetensors'] == 'Not installed':
        solutions.append(
            "📦 Install safetensors for safer checkpoint loading:\n"
            "   pip install safetensors"
        )
    
    solutions.append(
        "💾 Convert existing checkpoints to safetensors format:\n"
        "   from safetensors.torch import save_file\n"
        "   import torch\n"
        "   state_dict = torch.load('pytorch_model.bin')\n"
        "   save_file(state_dict, 'model.safetensors')"
    )
    
    solutions.append(
        "🆕 Start fresh training without checkpoint:\n"
        "   Set resume_from_checkpoint: null in your config"
    )
    
    return solutions


def main():
    print("🔍 Checking PyTorch and transformers compatibility...\n")
    
    # Check versions
    versions = check_versions()
    print("📋 Installed versions:")
    for package, version in versions.items():
        status = "✅" if version != 'Not installed' else "❌"
        print(f"   {status} {package}: {version}")
    
    print()
    
    # Check PyTorch compatibility
    torch_compatible, torch_msg = check_torch_compatibility(versions['torch'])
    status = "✅" if torch_compatible else "⚠️"
    print(f"{status} PyTorch compatibility: {torch_msg}")
    
    print()
    
    # Check checkpoint if provided
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"🔍 Checking checkpoint: {checkpoint_path}")
        
        checkpoint_info = check_checkpoint_format(checkpoint_path)
        if "error" in checkpoint_info:
            print(f"❌ {checkpoint_info['error']}")
        else:
            print(f"   📁 Path: {checkpoint_info['path']}")
            print(f"   📄 config.json: {'✅' if checkpoint_info['config_json'] else '❌'}")
            print(f"   🔧 pytorch_model.bin: {'✅' if checkpoint_info['pytorch_model_bin'] else '❌'}")
            print(f"   🛡️  safetensors files: {'✅' if checkpoint_info['has_safetensors'] else '❌'}")
            
            if checkpoint_info['has_safetensors']:
                print(f"      Found: {[f.name for f in checkpoint_info['safetensors_files']]}")
    
    print()
    
    # Suggest solutions if needed
    if not torch_compatible or versions['safetensors'] == 'Not installed':
        print("💡 Suggested solutions:")
        solutions = suggest_solutions(versions, torch_compatible)
        for i, solution in enumerate(solutions, 1):
            print(f"\n{i}. {solution}")
    else:
        print("✅ Your setup looks good!")
    
    print("\n" + "="*60)
    print("For more help, see: scripts/training/RESUME_TRAINING.md")


if __name__ == "__main__":
    main()
