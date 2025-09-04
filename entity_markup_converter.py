#!/usr/bin/env python3
"""
Entity Extraction Markup Converter

Converts JSONL entity extraction results into human-readable HTML format
for non-technical clients. Handles complex cases like overlapping spans,
multiple types per span, and repeated entities.

Usage:
    python entity_markup_converter.py input.jsonl output.html
"""

import json
import argparse
import html
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re


@dataclass
class EntitySpan:
    """Represents an entity span with its position and metadata."""
    start: int
    end: int
    text: str
    types: List[str]
    source: str  # 'ground_truth', 'mistral', 'gpt4'
    occurrence_id: int = 0  # For tracking repeated spans


@dataclass
class ComparisonResult:
    """Results of comparing entity annotations."""
    exact_matches: int = 0
    partial_matches: int = 0
    type_mismatches: int = 0
    ground_truth_only: int = 0
    model_only: int = 0
    total_ground_truth: int = 0
    total_model: int = 0


class EntityMarkupConverter:
    """Converts JSONL entity data to HTML markup."""
    
    def __init__(self):
        self.entity_colors = {
            'E21 Person': '#FF6B6B',
            'E53 Place': '#4ECDC4', 
            'E52 Time-Span': '#45B7D1',
            'E54 Dimension': '#96CEB4',
            'E19 Physical Thing': '#FFEAA7',
            'E74 Group': '#DDA0DD',
            'E86 Leaving': '#F39C12',
            'E9 Move': '#E17055',
            'F2 Expression': '#A29BFE',
            'E31 Document': '#FD79A8',
            'E55 Type': '#FDCB6E',
            'E7 Activity': '#6C5CE7',
            'default': '#BDC3C7'
        }
        
        self.source_labels = {
            'ground_truth': 'Ground Truth',
            'mistral': 'Mistral Small 3.2',
            'gpt4': 'GPT-4o Mini'
        }

    def find_all_occurrences(self, text: str, span_text: str) -> List[Tuple[int, int]]:
        """Find all occurrences of a span in the text."""
        occurrences = []
        start = 0
        while True:
            pos = text.find(span_text, start)
            if pos == -1:
                break
            occurrences.append((pos, pos + len(span_text)))
            start = pos + 1
        return occurrences

    def extract_entities(self, item: Dict) -> List[EntitySpan]:
        """Extract all entity spans from a JSONL item."""
        text = item['text']
        entities = []
        
        # Track occurrences of each span text
        span_counters = defaultdict(int)
        
        # Process ground truth
        for label in item.get('labels', []):
            span_text = label['span']
            types = label['types']
            occurrences = self.find_all_occurrences(text, span_text)
            
            for start, end in occurrences:
                span_counters[span_text] += 1
                entities.append(EntitySpan(
                    start=start,
                    end=end,
                    text=span_text,
                    types=types,
                    source='ground_truth',
                    occurrence_id=span_counters[span_text]
                ))
        
        # Process model outputs
        for model_key, source_name in [('mistral_small_3.2_output', 'mistral'), 
                                       ('gpt_4o_mini_output', 'gpt4')]:
            span_counters.clear()  # Reset for each model
            
            for prediction in item.get(model_key, []):
                span_text = prediction['span']
                types = prediction['types']
                occurrences = self.find_all_occurrences(text, span_text)
                
                for start, end in occurrences:
                    span_counters[span_text] += 1
                    entities.append(EntitySpan(
                        start=start,
                        end=end,
                        text=span_text,
                        types=types,
                        source=source_name,
                        occurrence_id=span_counters[span_text]
                    ))
        
        return sorted(entities, key=lambda x: (x.start, -x.end))

    def resolve_overlaps(self, entities: List[EntitySpan]) -> List[List[EntitySpan]]:
        """Group overlapping entities together."""
        if not entities:
            return []
        
        groups = []
        current_group = [entities[0]]
        
        for entity in entities[1:]:
            # Check if this entity overlaps with any in the current group
            overlaps = False
            for group_entity in current_group:
                if (entity.start < group_entity.end and 
                    entity.end > group_entity.start):
                    overlaps = True
                    break
            
            if overlaps:
                current_group.append(entity)
            else:
                groups.append(current_group)
                current_group = [entity]
        
        groups.append(current_group)
        return groups

    def get_primary_color(self, types: List[str]) -> str:
        """Get the primary color for an entity based on its types."""
        for entity_type in types:
            if entity_type in self.entity_colors:
                return self.entity_colors[entity_type]
        return self.entity_colors['default']
    
    def get_combined_colors(self, types: List[str]) -> Tuple[str, str]:
        """Get combined colors for multi-label entities."""
        if len(types) <= 1:
            primary = self.get_primary_color(types)
            return primary, primary
        
        # Get colors for all types
        colors = []
        for entity_type in types:
            if entity_type in self.entity_colors:
                colors.append(self.entity_colors[entity_type])
        
        if not colors:
            return self.entity_colors['default'], self.entity_colors['default']
        
        if len(colors) == 1:
            return colors[0], colors[0]
        
        # For multiple colors, create a gradient or pattern
        primary_color = colors[0]
        
        # Create a CSS gradient background for multiple types
        if len(colors) == 2:
            gradient = f"linear-gradient(45deg, {colors[0]} 50%, {colors[1]} 50%)"
        else:
            # For 3+ colors, create stripes
            stripe_width = 100 // len(colors)
            gradient_stops = []
            for i, color in enumerate(colors):
                start = i * stripe_width
                end = (i + 1) * stripe_width
                gradient_stops.append(f"{color} {start}%, {color} {end}%")
            gradient = f"linear-gradient(45deg, {', '.join(gradient_stops)})"
        
        return primary_color, gradient

    def format_types(self, types: List[str]) -> str:
        """Format entity types for display."""
        return ', '.join(types)

    def create_entity_html(self, entity: EntitySpan, is_primary: bool = True) -> str:
        """Create HTML markup for a single entity with sophisticated multi-label support."""
        primary_color, background = self.get_combined_colors(entity.types)
        opacity = '0.8' if is_primary else '0.4'
        
        # Create occurrence indicator if needed
        occurrence_suffix = f"<sup>{entity.occurrence_id}</sup>" if entity.occurrence_id > 1 else ""
        
        # Determine multi-label class and styling
        multi_label_class = ""
        css_variables = ""
        type_count_attr = ""
        
        if len(entity.types) > 1:
            multi_label_class = "multi-label"
            type_count_attr = f'data-type-count="{len(entity.types)}"'
            
            # Get colors for CSS variables
            colors = []
            for entity_type in entity.types:
                if entity_type in self.entity_colors:
                    colors.append(self.entity_colors[entity_type])
            
            if len(colors) >= 2:
                css_variables = f"--primary-color: {colors[0]}; --secondary-color: {colors[1]};"
                if len(colors) >= 3:
                    multi_label_class = "triple-label"
                    css_variables += f" --tertiary-color: {colors[2]};"
        
        # Create enhanced tooltip with structured content
        tooltip_html = f'''
        <div class="entity-tooltip">
            <div class="tooltip-section">
                <span class="tooltip-label">Source:</span> {self.source_labels[entity.source]}
            </div>
            <div class="tooltip-section">
                <span class="tooltip-label">Position:</span> {entity.start}-{entity.end}
            </div>
            <div class="tooltip-section">
                <span class="tooltip-label">Types:</span>
                <ul class="type-list">
        '''
        
        for entity_type in entity.types:
            color = self.entity_colors.get(entity_type, self.entity_colors['default'])
            tooltip_html += f'<li class="type-item" style="border-left: 3px solid {color};">{entity_type}</li>'
        
        tooltip_html += '''
                </ul>
            </div>
        </div>
        '''
        
        # Create data attributes for filtering
        types_attr = f'data-types="{"|".join(entity.types)}"'
        
        # Use gradient background for multi-type entities, solid color for single type
        if len(entity.types) > 1:
            background_style = f"background: {background};"
        else:
            background_style = f"background-color: {background};"
        
        return (
            f'<span class="entity {entity.source} {multi_label_class}" '
            f'style="{background_style} {css_variables} opacity: {opacity}; '
            f'border: 1px solid {primary_color};" '
            f'{types_attr} {type_count_attr}>'
            f'{html.escape(entity.text)}{occurrence_suffix}'
            f'{tooltip_html}'
            f'</span>'
        )

    def markup_text(self, text: str, entities: List[EntitySpan]) -> str:
        """Create HTML markup for text with entity highlighting."""
        if not entities:
            return html.escape(text)
        
        # Group overlapping entities
        overlap_groups = self.resolve_overlaps(entities)
        
        result = []
        last_pos = 0
        
        for group in overlap_groups:
            # Find the span that covers this group
            group_start = min(e.start for e in group)
            group_end = max(e.end for e in group)
            
            # Add text before this group
            if group_start > last_pos:
                result.append(html.escape(text[last_pos:group_start]))
            
            # Handle the group
            if len(group) == 1:
                # Single entity, simple case
                result.append(self.create_entity_html(group[0]))
            else:
                # Multiple overlapping entities
                # Use the longest span as primary, others as secondary
                primary = max(group, key=lambda x: x.end - x.start)
                secondaries = [e for e in group if e != primary]
                
                # Create nested markup
                primary_html = self.create_entity_html(primary)
                
                # Add secondary entities as additional info
                secondary_info = []
                for secondary in secondaries:
                    secondary_info.append(
                        f"{self.source_labels[secondary.source]}: "
                        f"{self.format_types(secondary.types)}"
                    )
                
                if secondary_info:
                    tooltip_extra = "\\nAlso annotated as:\\n" + "\\n".join(secondary_info)
                    primary_html = primary_html.replace(
                        'title="', f'title="{tooltip_extra}\\n'
                    )
                
                result.append(primary_html)
            
            last_pos = group_end
        
        # Add remaining text
        if last_pos < len(text):
            result.append(html.escape(text[last_pos:]))
        
        return ''.join(result)

    def compare_annotations(self, ground_truth: List[EntitySpan], 
                          model_entities: List[EntitySpan]) -> ComparisonResult:
        """Compare model annotations with ground truth."""
        result = ComparisonResult()
        
        # Convert to sets for comparison
        gt_spans = {(e.start, e.end, tuple(sorted(e.types))) for e in ground_truth}
        model_spans = {(e.start, e.end, tuple(sorted(e.types))) for e in model_entities}
        
        result.total_ground_truth = len(gt_spans)
        result.total_model = len(model_spans)
        result.exact_matches = len(gt_spans & model_spans)
        result.ground_truth_only = len(gt_spans - model_spans)
        result.model_only = len(model_spans - gt_spans)
        
        # Check for partial matches (same span, different types)
        gt_positions = {(e.start, e.end) for e in ground_truth}
        model_positions = {(e.start, e.end) for e in model_entities}
        
        overlapping_positions = gt_positions & model_positions
        exact_match_positions = {(s, e) for s, e, _ in gt_spans & model_spans}
        
        result.partial_matches = len(overlapping_positions - exact_match_positions)
        
        return result

    def generate_html_report(self, jsonl_file: str) -> str:
        """Generate complete HTML report from JSONL file."""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entity Extraction Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            padding-right: 320px; /* Make room for floating legend */
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        
        /* Floating Legend */
        .floating-legend {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 280px;
            background: white;
            border: 2px solid #3498db;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}
        
        .legend-header {{
            background: #3498db;
            color: white;
            padding: 12px 15px;
            margin: 0;
            border-radius: 8px 8px 0 0;
            font-size: 14px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .legend-toggle {{
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .legend-content {{
            padding: 15px;
            transition: all 0.3s ease;
        }}
        
        .legend-content.collapsed {{
            display: none;
        }}
        
        .legend-section {{
            margin-bottom: 20px;
        }}
        
        .legend-section:last-child {{
            margin-bottom: 0;
        }}
        
        .legend-section-title {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .legend-item:hover {{
            background: #f8f9fa;
            transform: translateX(2px);
        }}
        
        .legend-item.active {{
            background: #e3f2fd;
            border-left: 3px solid #2196f3;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 8px;
            border: 1px solid #ddd;
            flex-shrink: 0;
        }}
        
        .legend-color.multi-type {{
            background: linear-gradient(45deg, var(--color1) 50%, var(--color2) 50%);
        }}
        
        .legend-text {{
            flex: 1;
            line-height: 1.2;
        }}
        
        .source-legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .source-legend-item:hover {{
            background: #f8f9fa;
        }}
        
        .source-indicator {{
            width: 4px;
            height: 16px;
            margin-right: 8px;
            border-radius: 2px;
        }}
        
        .source-indicator.ground_truth {{ background: #27ae60; }}
        .source-indicator.mistral {{ background: #e74c3c; }}
        .source-indicator.gpt4 {{ background: #9b59b6; }}
        
        /* Entity Styles */
        .text-section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .source-section {{
            margin: 15px 0;
            padding: 15px;
            border-left: 4px solid #3498db;
            background: #f8f9fa;
        }}
        .source-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .text-content {{
            line-height: 2;
            font-size: 16px;
        }}
        
        .entity {{
            cursor: help;
            display: inline-block;
            position: relative;
            transition: all 0.2s ease;
            border-radius: 3px;
            padding: 2px 4px;
            margin: 1px;
        }}
        
        .entity:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 10;
        }}
        
        .entity.highlighted {{
            animation: pulse 1s ease-in-out;
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.6);
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7); }}
            70% {{ box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }}
        }}
        
        /* Multi-label entity styles */
        .entity.multi-label {{
            position: relative;
            background: linear-gradient(45deg, var(--primary-color) 0%, var(--primary-color) 50%, var(--secondary-color) 50%, var(--secondary-color) 100%);
        }}
        
        .entity.multi-label::after {{
            content: attr(data-type-count);
            position: absolute;
            top: -8px;
            right: -8px;
            background: #2c3e50;
            color: white;
            border-radius: 50%;
            width: 16px;
            height: 16px;
            font-size: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
        
        .entity.triple-label {{
            background: repeating-linear-gradient(
                45deg,
                var(--primary-color),
                var(--primary-color) 4px,
                var(--secondary-color) 4px,
                var(--secondary-color) 8px,
                var(--tertiary-color) 8px,
                var(--tertiary-color) 12px
            );
        }}
        
        /* Enhanced tooltip */
        .entity-tooltip {{
            position: absolute;
            background: #2c3e50;
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
        }}
        
        .entity:hover .entity-tooltip {{
            opacity: 1;
        }}
        
        .tooltip-section {{
            margin-bottom: 8px;
        }}
        
        .tooltip-section:last-child {{
            margin-bottom: 0;
        }}
        
        .tooltip-label {{
            font-weight: bold;
            color: #3498db;
        }}
        
        .type-list {{
            list-style: none;
            padding: 0;
            margin: 4px 0 0 0;
        }}
        
        .type-item {{
            padding: 2px 6px;
            margin: 2px 0;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            font-size: 11px;
        }}
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .stats-table th {{
            background-color: #3498db;
            color: white;
        }}
        .stats-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .summary {{
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .ground_truth {{ border-left: 4px solid #27ae60; }}
        .mistral {{ border-left: 4px solid #e74c3c; }}
        .gpt4 {{ border-left: 4px solid #9b59b6; }}
        
        /* Responsive design */
        @media (max-width: 1400px) {{
            body {{ padding-right: 20px; }}
            .floating-legend {{
                position: relative;
                top: auto;
                right: auto;
                width: 100%;
                margin: 20px 0;
            }}
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .floating-legend {{ width: 100%; }}
            .legend-content {{ padding: 10px; }}
        }}
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Toggle legend
            const toggleBtn = document.querySelector('.legend-toggle');
            const legendContent = document.querySelector('.legend-content');
            
            toggleBtn.addEventListener('click', function() {{
                legendContent.classList.toggle('collapsed');
                toggleBtn.textContent = legendContent.classList.contains('collapsed') ? '+' : '−';
            }});
            
            // Entity type filtering
            const entityTypeItems = document.querySelectorAll('.legend-item[data-type]');
            const sourceItems = document.querySelectorAll('.source-legend-item[data-source]');
            
            entityTypeItems.forEach(item => {{
                item.addEventListener('click', function() {{
                    const type = this.dataset.type;
                    const isActive = this.classList.contains('active');
                    
                    // Clear all active states
                    entityTypeItems.forEach(i => i.classList.remove('active'));
                    
                    if (!isActive) {{
                        this.classList.add('active');
                        highlightEntitiesByType(type);
                    }} else {{
                        clearHighlights();
                    }}
                }});
            }});
            
            sourceItems.forEach(item => {{
                item.addEventListener('click', function() {{
                    const source = this.dataset.source;
                    const isActive = this.classList.contains('active');
                    
                    // Clear all active states
                    sourceItems.forEach(i => i.classList.remove('active'));
                    
                    if (!isActive) {{
                        this.classList.add('active');
                        highlightEntitiesBySource(source);
                    }} else {{
                        clearHighlights();
                    }}
                }});
            }});
            
            function highlightEntitiesByType(type) {{
                clearHighlights();
                const entities = document.querySelectorAll('.entity');
                entities.forEach(entity => {{
                    const entityTypes = entity.getAttribute('data-types');
                    if (entityTypes && entityTypes.includes(type)) {{
                        entity.classList.add('highlighted');
                    }} else {{
                        entity.style.opacity = '0.3';
                    }}
                }});
            }}
            
            function highlightEntitiesBySource(source) {{
                clearHighlights();
                const entities = document.querySelectorAll('.entity');
                entities.forEach(entity => {{
                    if (entity.classList.contains(source)) {{
                        entity.classList.add('highlighted');
                    }} else {{
                        entity.style.opacity = '0.3';
                    }}
                }});
            }}
            
            function clearHighlights() {{
                const entities = document.querySelectorAll('.entity');
                entities.forEach(entity => {{
                    entity.classList.remove('highlighted');
                    entity.style.opacity = '';
                }});
            }}
        }});
    </script>
</head>
<body>
    <div class="container">
        <h1>Entity Extraction Comparison Report</h1>
        <p>This report compares entity extraction results between ground truth annotations and two language models: Mistral Small 3.2 and GPT-4o Mini.</p>
        
        <div class="legend">
            <h3>Entity Type Legend</h3>
"""
        
        # Create floating legend
        html_content += """
        <!-- Floating Legend -->
        <div class="floating-legend">
            <div class="legend-header">
                <span>Entity Legend</span>
                <button class="legend-toggle">−</button>
            </div>
            <div class="legend-content">
                <div class="legend-section">
                    <div class="legend-section-title">Entity Types</div>
"""
        
        # Add entity type legend items
        for entity_type, color in self.entity_colors.items():
            if entity_type != 'default':
                html_content += f'''
                    <div class="legend-item" data-type="{entity_type}">
                        <div class="legend-color" style="background-color: {color};"></div>
                        <div class="legend-text">{entity_type}</div>
                    </div>'''
        
        html_content += """
                </div>
                <div class="legend-section">
                    <div class="legend-section-title">Sources</div>
                    <div class="source-legend-item" data-source="ground_truth">
                        <div class="source-indicator ground_truth"></div>
                        <div class="legend-text">Ground Truth</div>
                    </div>
                    <div class="source-legend-item" data-source="mistral">
                        <div class="source-indicator mistral"></div>
                        <div class="legend-text">Mistral Small 3.2</div>
                    </div>
                    <div class="source-legend-item" data-source="gpt4">
                        <div class="source-indicator gpt4"></div>
                        <div class="legend-text">GPT-4o Mini</div>
                    </div>
                </div>
                <div class="legend-section">
                    <div class="legend-section-title">Multi-Label</div>
                    <div class="legend-item">
                        <div class="legend-color multi-type" style="--color1: #FF6B6B; --color2: #4ECDC4;"></div>
                        <div class="legend-text">Multiple Types</div>
                    </div>
                </div>
            </div>
        </div>
"""
        
        # Process JSONL file
        total_comparisons = {'mistral': ComparisonResult(), 'gpt4': ComparisonResult()}
        text_count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line)
                    text_count += 1
                    
                    # Extract entities
                    all_entities = self.extract_entities(item)
                    ground_truth = [e for e in all_entities if e.source == 'ground_truth']
                    mistral_entities = [e for e in all_entities if e.source == 'mistral']
                    gpt4_entities = [e for e in all_entities if e.source == 'gpt4']
                    
                    # Add to HTML
                    html_content += f"""
        <div class="text-section">
            <h3>Text Passage {text_count}</h3>
            
            <div class="source-section ground_truth">
                <div class="source-title">Ground Truth ({len(ground_truth)} entities)</div>
                <div class="text-content">{self.markup_text(item['text'], ground_truth)}</div>
            </div>
            
            <div class="source-section mistral">
                <div class="source-title">Mistral Small 3.2 ({len(mistral_entities)} entities)</div>
                <div class="text-content">{self.markup_text(item['text'], mistral_entities)}</div>
            </div>
            
            <div class="source-section gpt4">
                <div class="source-title">GPT-4o Mini ({len(gpt4_entities)} entities)</div>
                <div class="text-content">{self.markup_text(item['text'], gpt4_entities)}</div>
            </div>
"""
                    
                    # Compare annotations
                    mistral_comparison = self.compare_annotations(ground_truth, mistral_entities)
                    gpt4_comparison = self.compare_annotations(ground_truth, gpt4_entities)
                    
                    # Add comparison stats
#                     html_content += f"""
#             <table class="stats-table">
#                 <tr>
#                     <th>Metric</th>
#                     <th>Mistral Small 3.2</th>
#                     <th>GPT-4o Mini</th>
#                 </tr>
#                 <tr>
#                     <td>Exact Matches</td>
#                     <td>{mistral_comparison.exact_matches}</td>
#                     <td>{gpt4_comparison.exact_matches}</td>
#                 </tr>
#                 <tr>
#                     <td>Partial Matches</td>
#                     <td>{mistral_comparison.partial_matches}</td>
#                     <td>{gpt4_comparison.partial_matches}</td>
#                 </tr>
#                 <tr>
#                     <td>Model Only</td>
#                     <td>{mistral_comparison.model_only}</td>
#                     <td>{gpt4_comparison.model_only}</td>
#                 </tr>
#                 <tr>
#                     <td>Ground Truth Only</td>
#                     <td>{mistral_comparison.ground_truth_only}</td>
#                     <td>{gpt4_comparison.ground_truth_only}</td>
#                 </tr>
#                 <tr>
#                     <td>Precision</td>
#                     <td>{mistral_comparison.exact_matches / max(mistral_comparison.total_model, 1):.2%}</td>
#                     <td>{gpt4_comparison.exact_matches / max(gpt4_comparison.total_model, 1):.2%}</td>
#                 </tr>
#                 <tr>
#                     <td>Recall</td>
#                     <td>{mistral_comparison.exact_matches / max(mistral_comparison.total_ground_truth, 1):.2%}</td>
#                     <td>{gpt4_comparison.exact_matches / max(gpt4_comparison.total_ground_truth, 1):.2%}</td>
#                 </tr>
#             </table>
#         </div>
# """
                    html_content += """
         </div>
"""
                    
                    # Accumulate totals
                    for attr in ['exact_matches', 'partial_matches', 'ground_truth_only', 
                               'model_only', 'total_ground_truth', 'total_model']:
                        setattr(total_comparisons['mistral'], attr, 
                               getattr(total_comparisons['mistral'], attr) + getattr(mistral_comparison, attr))
                        setattr(total_comparisons['gpt4'], attr,
                               getattr(total_comparisons['gpt4'], attr) + getattr(gpt4_comparison, attr))
                
                except json.JSONDecodeError as e:
                    html_content += f'<p style="color: red;">Error parsing line {line_num}: {e}</p>\n'
                except Exception as e:
                    html_content += f'<p style="color: red;">Error processing line {line_num}: {e}</p>\n'
        
        # Add overall summary
        mistral_total = total_comparisons['mistral']
        gpt4_total = total_comparisons['gpt4']
        
        html_content += f"""
        <div class="summary">
            <h2>Overall Summary</h2>
            <p>Processed {text_count} text passages</p>
            
            <table class="stats-table">
                <tr>
                    <th>Overall Metrics</th>
                    <th>Mistral Small 3.2</th>
                    <th>GPT-4o Mini</th>
                </tr>
                <tr>
                    <td>Total Entities Predicted</td>
                    <td>{mistral_total.total_model}</td>
                    <td>{gpt4_total.total_model}</td>
                </tr>
                <tr>
                    <td>Total Ground Truth Entities</td>
                    <td colspan="2">{mistral_total.total_ground_truth}</td>
                </tr>
                <tr>
                    <td>Exact Matches</td>
                    <td>{mistral_total.exact_matches}</td>
                    <td>{gpt4_total.exact_matches}</td>
                </tr>
                <tr>
                    <td>Overall Precision</td>
                    <td>{mistral_total.exact_matches / max(mistral_total.total_model, 1):.2%}</td>
                    <td>{gpt4_total.exact_matches / max(gpt4_total.total_model, 1):.2%}</td>
                </tr>
                <tr>
                    <td>Overall Recall</td>
                    <td>{mistral_total.exact_matches / max(mistral_total.total_ground_truth, 1):.2%}</td>
                    <td>{gpt4_total.exact_matches / max(gpt4_total.total_ground_truth, 1):.2%}</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td>{2 * (mistral_total.exact_matches / max(mistral_total.total_model, 1)) * (mistral_total.exact_matches / max(mistral_total.total_ground_truth, 1)) / max((mistral_total.exact_matches / max(mistral_total.total_model, 1)) + (mistral_total.exact_matches / max(mistral_total.total_ground_truth, 1)), 0.001):.2%}</td>
                    <td>{2 * (gpt4_total.exact_matches / max(gpt4_total.total_model, 1)) * (gpt4_total.exact_matches / max(gpt4_total.total_ground_truth, 1)) / max((gpt4_total.exact_matches / max(gpt4_total.total_model, 1)) + (gpt4_total.exact_matches / max(gpt4_total.total_ground_truth, 1)), 0.001):.2%}</td>
                </tr>
            </table>
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3>How to Read This Report</h3>
            <ul>
                <li><strong>Highlighted text</strong> shows identified entities with color coding by type</li>
                <li><strong>Hover over entities</strong> to see detailed information</li>
                <li><strong>Superscript numbers</strong> indicate repeated occurrences of the same entity</li>
                <li><strong>Exact Matches</strong>: Same text span and same entity types</li>
                <li><strong>Partial Matches</strong>: Same text span but different entity types</li>
                <li><strong>Model Only</strong>: Entities found by the model but not in ground truth</li>
                <li><strong>Ground Truth Only</strong>: Entities in ground truth but missed by the model</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content

    def convert_file(self, input_file: str, output_file: str):
        """Convert JSONL file to HTML report."""
        print(f"Converting {input_file} to {output_file}...")
        
        try:
            html_content = self.generate_html_report(input_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Successfully created HTML report: {output_file}")
            print(f"📊 Open the file in your web browser to view the results")
            
        except Exception as e:
            print(f"❌ Error converting file: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL entity extraction results to HTML report"
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output HTML file")
    
    args = parser.parse_args()
    
    converter = EntityMarkupConverter()
    converter.convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
