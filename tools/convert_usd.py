# convert_usd.py
from pxr import Usd
import sys
import os

def convert_usd_binary_to_ascii(input_file, output_file):
    """Convert binary USD (.usdc) to ASCII USD (.usda)"""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return False
    
    try:
        # Open the binary USD stage
        stage = Usd.Stage.Open(input_file)
        if not stage:
            print(f"Error: Could not open USD file '{input_file}'")
            return False
        
        # Export to ASCII format
        success = stage.Export(output_file)
        if success:
            print(f"Successfully converted '{input_file}' to '{output_file}'")
            return True
        else:
            print(f"Error: Failed to export to '{output_file}'")
            return False
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_usd.py input.usdc output.usda")
        print("Example: python convert_usd.py my_asset.usdc my_asset.usda")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_usd_binary_to_ascii(input_file, output_file)
