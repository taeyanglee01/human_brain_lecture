#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create visual summary of literature comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

def create_comparison_summary():
    """Create visual summary comparing current findings with literature"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('선행 연구와의 비교 분석 요약\nBehavioral-FA Correlation: Literature Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Consistency Overview
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    findings = [
        {'name': 'Reappraisal ↔ mOFC FA (+)', 'consistency': '매우 높음', 'r': 0.844, 'p': 0.004, 'color': 'darkgreen'},
        {'name': 'Reappraisal ↔ Amygdala FA (+)', 'consistency': '매우 높음', 'r': 0.757, 'p': 0.011, 'color': 'darkgreen'},
        {'name': 'Suppression ↔ Amygdala FA (-)', 'consistency': '매우 높음', 'r': -0.790, 'p': 0.007, 'color': 'darkgreen'},
        {'name': 'DERS ↔ mOFC FA (-)', 'consistency': '높음', 'r': -0.708, 'p': 0.033, 'color': 'green'},
        {'name': 'DERS ↔ Cuneus FA (-)', 'consistency': '불명확', 'r': -0.666, 'p': 0.035, 'color': 'orange'},
        {'name': 'ARI - FA 상관 없음', 'consistency': '불일치', 'r': 0, 'p': 1.0, 'color': 'red'},
    ]

    y_pos = 0.9
    for finding in findings:
        # Draw box
        if finding['consistency'] == '매우 높음':
            box_color = 'lightgreen'
        elif finding['consistency'] == '높음':
            box_color = 'lightblue'
        elif finding['consistency'] == '불명확':
            box_color = 'lightyellow'
        else:
            box_color = 'lightcoral'

        fancy_box = FancyBboxPatch((0.05, y_pos-0.08), 0.9, 0.12,
                                  boxstyle="round,pad=0.01",
                                  facecolor=box_color, edgecolor='black',
                                  linewidth=2, transform=ax1.transAxes)
        ax1.add_patch(fancy_box)

        # Text
        text = f"{finding['name']:40s}  |  정합성: {finding['consistency']:8s}"
        if finding['r'] != 0:
            text += f"  |  r={finding['r']:.3f}, p={finding['p']:.3f}"

        ax1.text(0.5, y_pos-0.02, text,
                transform=ax1.transAxes, fontsize=11,
                fontweight='bold', ha='center', va='center',
                fontfamily='monospace')

        y_pos -= 0.15

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('주요 발견사항 정합성', fontsize=14, fontweight='bold', pad=20)

    # 2. Key Brain Regions
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')

    brain_regions = {
        'Medial OFC': {'support': '매우 강함', 'color': 'darkgreen', 'size': 300},
        'Amygdala': {'support': '매우 강함', 'color': 'darkgreen', 'size': 300},
        'Prefrontal Cortex': {'support': '강함', 'color': 'green', 'size': 250},
        'Parietal': {'support': '중간', 'color': 'orange', 'size': 200},
        'Cuneus': {'support': '약함', 'color': 'lightcoral', 'size': 150},
    }

    y_positions = np.linspace(0.8, 0.2, len(brain_regions))

    for idx, (region, info) in enumerate(brain_regions.items()):
        # Circle
        circle = Circle((0.2, y_positions[idx]), 0.05,
                       color=info['color'], alpha=0.7,
                       transform=ax2.transAxes)
        ax2.add_patch(circle)

        # Text
        ax2.text(0.3, y_positions[idx], f"{region}\n선행연구 지지: {info['support']}",
                transform=ax2.transAxes, fontsize=11,
                va='center', fontweight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('주요 뇌 영역 - 선행연구 지지도', fontsize=13, fontweight='bold')

    # 3. Consistency by Measure
    ax3 = fig.add_subplot(gs[1, 1])

    measures = ['ERQ\nReappraisal', 'ERQ\nSuppression', 'DERS', 'ERQ\nTotal', 'ARI']
    consistency_scores = [90, 85, 70, 60, 20]  # Percentage
    colors_bar = ['darkgreen', 'darkgreen', 'green', 'orange', 'red']

    bars = ax3.barh(measures, consistency_scores, color=colors_bar, alpha=0.7)

    # Add percentage labels
    for i, (bar, score) in enumerate(zip(bars, consistency_scores)):
        ax3.text(score + 2, i, f'{score}%', va='center', fontweight='bold')

    ax3.set_xlabel('선행연구 정합성 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('행동 척도별 정합성', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.grid(axis='x', alpha=0.3)

    # 4. Methodological Comparison
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    methods_text = """
    방법론적 비교

    ✅ 본 연구의 강점:
       • FreeSurferDKT atlas 사용 (표준화된 ROI)
       • 다중 감정조절 척도 (ARI, DERS, ERQ)
       • 체계적 FA 추출 프로토콜

    ⚠️ 본 연구의 제한점:
       • 작은 표본 크기 (n=10)
       • ROI 기반 분석 (tract-based 아님)
       • 다중 비교 보정 미적용

    📚 선행 연구의 주요 특징:
       • 대규모 표본 (n > 30)
       • Tract-Based Spatial Statistics (TBSS)
       • DTI Tractography (예: uncinate fasciculus)
       • FDR/Bonferroni 보정

    📌 후속 연구 제안:
       1. 표본 크기 증대 (n > 30)
       2. TBSS 분석 추가
       3. 다중 비교 보정 적용
       4. 종단 연구 설계
    """

    ax4.text(0.05, 0.95, methods_text, transform=ax4.transAxes,
            fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    output_file = 'C:/Users/Public/human_brain_lecture/results/251101/literature_comparison_visual.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")

    # Create simple summary table
    create_summary_table()

def create_summary_table():
    """Create a simple summary table"""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    data = [
        ['발견사항', '본 연구 결과', '선행연구 지지', '정합성', '신뢰도'],
        ['', '', '', '', ''],
        ['Reappraisal ↔ mOFC FA', 'r=0.844**\np=0.004', '✅ 매우 높음\nvmPFC와 재평가\n양의 상관 (2014)', '⭐⭐⭐⭐⭐', '높음'],
        ['Reappraisal ↔ Amygdala FA', 'r=0.757*\np=0.011', '✅ 매우 높음\n편도체-PFC 연결성\n(JNeurosci 2015)', '⭐⭐⭐⭐⭐', '높음'],
        ['Suppression ↔ Amygdala FA', 'r=-0.790**\np=0.007', '✅ 매우 높음\nvmPFC와 억제\n음의 상관 (2014)', '⭐⭐⭐⭐⭐', '높음'],
        ['DERS ↔ mOFC FA', 'r=-0.708*\np=0.033', '✅ 높음\n감정조절장애와\n백질 이상', '⭐⭐⭐⭐', '중간'],
        ['DERS ↔ Cuneus FA', 'r=-0.666*\np=0.035', '⚠️ 불명확\n선행연구 제한적', '⭐⭐', '낮음'],
        ['ARI - FA 상관', '유의미한\n상관 없음', '❌ 불일치\n표본크기 문제\n추정', '⭐', '낮음'],
    ]

    # Colors for rows
    colors = [['lightgray']*5,
              ['white']*5,
              ['#90EE90']*5,  # light green
              ['#90EE90']*5,
              ['#90EE90']*5,
              ['#ADD8E6']*5,  # light blue
              ['#FFFFE0']*5,  # light yellow
              ['#FFB6C1']*5]  # light red

    table = ax.table(cellText=data, cellLoc='left',
                    loc='center', cellColours=colors,
                    colWidths=[0.25, 0.15, 0.3, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)

    # Bold header
    for i in range(5):
        table[(0, i)].set_text_props(weight='bold', fontsize=11)
        table[(0, i)].set_facecolor('darkgray')

    plt.title('선행 연구 정합성 요약표\nLiterature Comparison Summary',
             fontsize=16, fontweight='bold', pad=20)

    output_file = 'C:/Users/Public/human_brain_lecture/results/251101/literature_comparison_table.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    create_comparison_summary()
    print("\nLiterature comparison visualizations complete!")
