#!/usr/bin/env python3
# ><(((o>  只读取图像元数据，不把文件修改时间当作采集时间。
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/path/to/source_fish_landmark_data')
FIELDS = {271: 'make', 272: 'model', 306: 'image_datetime',
          36867: 'datetime_original', 36868: 'datetime_digitized',
          42036: 'lens_model', 305: 'software'}


def inspect(path):
    result = {name: '' for name in FIELDS.values()}
    result.update(error='', exif_tag_count=0)
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            tags = dict(exif)
            if 34665 in exif:
                tags.update(exif.get_ifd(34665))
            result['exif_tag_count'] = len(tags)
            for key, name in FIELDS.items():
                value = tags.get(key, '')
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                result[name] = str(value).strip('\x00 ')
    except Exception as exc:
        result['error'] = str(exc)
    return result


def main():
    metadata = pd.read_parquet(ROOT / 'data/derived/metadata.parquet')
    sources = sorted(set(metadata.original_image))
    rows = []
    for cohort, paths in [('source', [SOURCE / x for x in sources]),
                          ('release', sorted((ROOT / 'data/benchmark_v1/images/original').glob('*')))]:
        for path in paths:
            if path.is_file():
                rows.append(dict(cohort=cohort, path=str(path.relative_to(SOURCE) if cohort == 'source' else path.relative_to(ROOT)), **inspect(path)))
            else:
                rows.append(dict(cohort=cohort, path=str(path), **{**inspect(path), 'error': 'missing file'}))
    output = ROOT / 'qa/exif-audit-0911.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'source_records': len(metadata), 'source_unique_paths': len(sources), 'cohorts': {}}
    for cohort in ['source', 'release']:
        part = [r for r in rows if r['cohort'] == cohort]
        summary['cohorts'][cohort] = dict(images=len(part), errors=sum(bool(r['error']) for r in part),
            images_with_exif=sum(r['exif_tag_count'] > 0 for r in part),
            fields={name: dict(Counter(r[name] for r in part if r[name])) for name in FIELDS.values()})
    (ROOT / 'qa/exif-audit-0911.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
