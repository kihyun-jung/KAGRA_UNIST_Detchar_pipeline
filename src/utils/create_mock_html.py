#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mock Hveto HTML Generator
=========================

Hveto 분석 결과 페이지(index.html)를 표준 출력 경로에 생성합니다.
Target Path: results/hveto/{DATE}/index.html
"""

import os
from pathlib import Path

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def generate_mock_index_html(date_str):
    # 1. Hveto 결과 폴더 경로 설정 (사용자 요청 반영)
    # results/hveto/2023-06-18/
    hveto_result_dir = BASE_DIR / "results" / "hveto" / date_str
    
    # 폴더가 없으면 생성 (Hveto 파이프라인이 안 돌았을 경우를 대비)
    hveto_result_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = hveto_result_dir / "index.html"
    
    # 2. Mock HTML 내용 (박사님이 주신 구조 기반)
    html_content = f"""
<!DOCTYPE HTML>
<html lang="en">
<head><title>K1 Hveto | {date_str}</title></head>
<body>
<div class="container">
<h2 class="mt-4">Summary</h2>
<table class="table table-sm table-hover">
<thead>
<tr>
<th scope="row">Round</th>
<th scope="row">Winner</th>
<th scope="row">Twin [s]</th>
<th scope="row">SNR Thresh</th>
<th scope="row">Significance</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>K1:PEM-MOCK_SENSOR_1</td>
<td>0.40</td>
<td>40.00</td>
<td>15.57</td>
</tr>
<tr>
<td>2</td>
<td>K1:PEM-MOCK_SENSOR_2</td>
<td>0.40</td>
<td>8.00</td>
<td>15.18</td>
</tr>
<tr>
<td>3</td>
<td>K1:PEM-MOCK_SENSOR_3</td>
<td>1.00</td>
<td>7.75</td>
<td>10.96</td>
</tr>
<tr>
<td>4</td>
<td>K1:PEM-MOCK_SENSOR_4</td>
<td>0.40</td>
<td>20.00</td>
<td>6.70</td>
</tr>
<tr>
<td>5</td>
<td>K1:PEM-MOCK_SENSOR_5</td>
<td>0.40</td>
<td>100.00</td>
<td>5.89</td>
</tr>
</tbody>
</table>
</div>
</body>
</html>
    """
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[*] Created mock HTML at: {html_path}")

if __name__ == "__main__":
    # 테스트용 날짜
    generate_mock_index_html("2023-06-18")
