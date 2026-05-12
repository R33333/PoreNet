"""
轻量级 AM 文件检查（不加载数据，内存友好）
"""
import os
import re

def check_am_header(am_path):
    """只检查 AM 文件头部，不加载数据"""
    print('='*50)
    print('AM FILE HEADER CHECK (Low Memory)')
    print('='*50)
    
    print(f'\nFile: {os.path.basename(am_path)}')
    print(f'File size: {os.path.getsize(am_path) / 1024 / 1024:.2f} MB')
    
    # 只读取头部（前 2KB）
    with open(am_path, 'rb') as f:
        header = f.read(2000).decode('ascii', errors='ignore')
    
    print('\n--- Header Content (first 1000 chars) ---')
    print(header[:1000])
    
    # 解析关键信息
    print('\n--- Parsed Info ---')
    
    # 维度
    match = re.search(r'define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)', header)
    if match:
        nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))
        print(f'Dimensions: {nx} x {ny} x {nz}')
        print(f'Total voxels: {nx*ny*nz:,}')
        print(f'Expected size (uint8): {nx*ny*nz / 1024 / 1024:.2f} MB')
    else:
        print('WARNING: Could not parse dimensions!')
        return False
    
    # 压缩信息
    rle_match = re.search(r'HxByteRLE,(\d+)', header)
    if rle_match:
        rle_size = int(rle_match.group(1))
        print(f'RLE compressed size: {rle_size / 1024 / 1024:.2f} MB')
        print(f'File compression ratio: {nx*ny*nz / rle_size:.1f}x')
    
    # 数据类型
    if 'byte Labels' in header:
        print('Data type: byte Labels (segmented data)')
    elif 'byte Data' in header:
        print('Data type: byte Data (grayscale)')
    
    # 材料信息
    materials = re.findall(r'Material\d+\s*\{[^}]*Id\s+(\d+)', header)
    if materials:
        print(f'Materials found: {len(materials)} phases')
        print(f'Material IDs: {materials}')
    
    print('\n' + '='*50)
    print('RESULT: AM file format is VALID!')
    print('='*50)
    return True


if __name__ == '__main__':
    # 检查两个数据文件
    data_files = [
        r"D:\code\inr_digital_rock\DRP-265\Berea Altered_ mixed-wet\Berea_Altered_waterflooding_segmented\Berea_Altered_waterflooding_segmented.am",
        r"D:\code\inr_digital_rock\DRP-265\Berea Unaltered_ water-wet\Berea_Unaltered_waterflooding_segmented\Berea_Unaltered_waterflooding_segmented.am"
    ]
    
    for am_path in data_files:
        if os.path.exists(am_path):
            print(f'\n\nChecking: {os.path.basename(am_path)}\n')
            check_am_header(am_path)
        else:
            print(f'File not found: {am_path}')
