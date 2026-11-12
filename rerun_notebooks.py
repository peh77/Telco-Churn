"""분석 노트북 순차 실행 및 출력·그래프 저장."""
from pathlib import Path
import base64
import json
import shutil
from datetime import datetime

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'rerun'
OUT.mkdir(parents=True, exist_ok=True)
backup = OUT / ('previous_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir()
for path in (ROOT / 'results').iterdir():
    if path.is_file():
        shutil.copy2(path, backup / path.name)

for path in sorted(ROOT.glob('Telco_Churn_STEP*.ipynb')):
    print('START', path.name, flush=True)
    nb = nbformat.read(path, as_version=4)
    # 한글 글꼴과 인라인 그래프 출력을 설정한다.
    nb.cells.insert(0, nbformat.v4.new_code_cell(
        "%matplotlib inline\nimport matplotlib.pyplot as plt\n"
        "plt.rcParams['font.family'] = 'Malgun Gothic'\n"
        "plt.rcParams['axes.unicode_minus'] = False"
    ))
    client = NotebookClient(nb, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}})
    destination = OUT / path.name
    try:
        client.execute()
    finally:
        nbformat.write(nb, destination)
    text_parts = []
    image_count = 0
    for index, cell in enumerate(nb.cells):
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                text_parts.append(output.text)
            data = output.get('data', {})
            if 'text/plain' in data:
                text_parts.append(data['text/plain'])
            if 'image/png' in data:
                image_count += 1
                image_path = OUT / f'{path.stem}_figure_{image_count:02d}.png'
                image_path.write_bytes(base64.b64decode(data['image/png']))
    (OUT / f'{path.stem}_outputs.txt').write_text('\n\n'.join(text_parts), encoding='utf-8')
    print('DONE', path.name, 'figures:', image_count, flush=True)
print('ALL NOTEBOOKS COMPLETED', flush=True)
