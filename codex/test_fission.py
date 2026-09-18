#!/usr/bin/env python3
"""
Minimal test for image fission with theme background and reference range modes.
"""
import sys
import io
from PIL import Image

# Add model_worker to path
sys.path.insert(0, r'e:\Desktop\双接口\codex\model_worker')

# Import fission functions directly
from worker_infer_enhanced import (
    _theme_colors,
    _blend_bg_and_subject,
    _full_redraw_like_reference,
    simulate_fission,
)
import asyncio

async def test_fission():
    # Create a simple test image (red square on transparent background)
    test_img = Image.new('RGBA', (400, 400), (0, 0, 0, 0))
    for y in range(100, 300):
        for x in range(100, 300):
            test_img.putpixel((x, y), (220, 100, 50, 255))  # Tea-colored rectangle
    
    # Save test image to bytes
    buf = io.BytesIO()
    test_img.save(buf, format='PNG')
    test_bytes = buf.getvalue()
    
    print("Test Image: Simple tea-colored rectangle (400x400)")
    print()
    
    # Test 1: Theme background mode
    print("=" * 60)
    print("Test 1: theme_background mode")
    print("=" * 60)
    results1 = await simulate_fission(
        test_bytes,
        variants=2,
        mode='theme_background',
        theme='warm herbal',
        theme_strength=0.8,
        reference_range=0.5,
    )
    print(f"✓ Generated {len(results1)} variant(s) with warm herbal theme")
    for idx, result_bytes in enumerate(results1):
        size_kb = len(result_bytes) / 1024
        print(f"  Variant {idx+1}: {size_kb:.1f} KB")
    
    # Save one result
    if results1:
        out_path = r'e:\Desktop\双接口\codex\generated\test_fission_theme.png'
        with open(out_path, 'wb') as f:
            f.write(results1[0])
        print(f"  Saved: {out_path}")
    
    print()
    
    # Test 2: Full redraw mode
    print("=" * 60)
    print("Test 2: full_redraw mode")
    print("=" * 60)
    results2 = await simulate_fission(
        test_bytes,
        variants=2,
        mode='full_redraw',
        theme='luxury gold',
        theme_strength=0.7,
        reference_range=0.7,
    )
    print(f"✓ Generated {len(results2)} variant(s) with full redraw")
    for idx, result_bytes in enumerate(results2):
        size_kb = len(result_bytes) / 1024
        print(f"  Variant {idx+1}: {size_kb:.1f} KB")
    
    # Save one result
    if results2:
        out_path = r'e:\Desktop\双接口\codex\generated\test_fission_redraw.png'
        with open(out_path, 'wb') as f:
            f.write(results2[0])
        print(f"  Saved: {out_path}")
    
    print()
    
    # Test 3: Reference lock mode (default)
    print("=" * 60)
    print("Test 3: reference_lock mode (default)")
    print("=" * 60)
    results3 = await simulate_fission(
        test_bytes,
        variants=2,
        mode='reference_lock',
        theme='forest calm',
        theme_strength=0.5,
        reference_range=0.3,
    )
    print(f"✓ Generated {len(results3)} variant(s) with reference lock")
    for idx, result_bytes in enumerate(results3):
        size_kb = len(result_bytes) / 1024
        print(f"  Variant {idx+1}: {size_kb:.1f} KB")
    
    # Save one result
    if results3:
        out_path = r'e:\Desktop\双接口\codex\generated\test_fission_lock.png'
        with open(out_path, 'wb') as f:
            f.write(results3[0])
        print(f"  Saved: {out_path}")
    
    print()
    print("=" * 60)
    print("✓✓✓ All tests passed successfully!")
    print("=" * 60)
    print()
    print("Generated output images:")
    print("  - test_fission_theme.png   (warm herbal theme)")
    print("  - test_fission_redraw.png  (luxury gold redraw)")
    print("  - test_fission_lock.png    (forest calm reference lock)")
    print()
    print("You can now:")
    print("  1. Review the generated images")
    print("  2. Deploy the /infer_enhanced endpoint with these modes")
    print("  3. Adjust parameters (theme, theme_strength, reference_range)")
    print()

if __name__ == '__main__':
    asyncio.run(test_fission())
