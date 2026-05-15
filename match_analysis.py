import json
import sqlite3
from pathlib import Path

# Load GT entities
gt_entities = []
for cr_file in ['ground_truth/calibration/cr01.json', 'ground_truth/calibration/cr02.json', 
                 'ground_truth/calibration/cr03.json', 'ground_truth/calibration/cr04.json', 
                 'ground_truth/calibration/cr05.json']:
    with open(cr_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entity in data.get('impacted_entities', []):
            gt_entities.append({
                'cr': data.get('cr_id'),
                'node': entity.get('node'),
                'justification': entity.get('justification')
            })

print(f'Loaded {len(gt_entities)} GT entities')

# Load AST nodes
conn = sqlite3.connect('data/impactracer.db')
cursor = conn.cursor()
cursor.execute('SELECT node_id, node_type, file_path FROM code_nodes')
ast_rows = cursor.fetchall()

# Build lookup structures
node_id_set = set(row[0] for row in ast_rows)
file_nodes = {}
for node_id, node_type, file_path in ast_rows:
    if file_path not in file_nodes:
        file_nodes[file_path] = []
    file_nodes[file_path].append((node_id, node_type))

indexed_files = set(file_path for _, _, file_path in ast_rows)

print(f'AST has {len(ast_rows)} nodes in {len(indexed_files)} files')
print()

# Classify each GT entity
results = {
    'EXACT_MATCH': [],
    'FILE_ONLY_MATCH': [],
    'NO_FILE_MATCH': [],
    'NODE_TYPE_GAP': []
}

for gt in gt_entities:
    node_str = gt['node'].strip()
    
    # Parse node string
    if '::' in node_str:
        file_path, symbol = node_str.rsplit('::', 1)
    else:
        file_path = node_str
        symbol = None
    
    # Check if exact match exists
    if node_str in node_id_set:
        results['EXACT_MATCH'].append(gt)
    elif symbol:
        if file_path in indexed_files:
            results['FILE_ONLY_MATCH'].append(gt)
        else:
            results['NO_FILE_MATCH'].append(gt)
    else:
        if file_path in indexed_files:
            results['FILE_ONLY_MATCH'].append(gt)
        else:
            results['NO_FILE_MATCH'].append(gt)

# Report
print('=== CLASSIFICATION ===')
for category, items in results.items():
    print(f'{category}: {len(items)}')

print()
print('=== EXACT MATCHES ===')
for item in results['EXACT_MATCH']:
    print(f'  {item["cr"]} | {item["node"]}')

print()
print('=== FILE_ONLY_MATCH (Symbol not found in indexed file) ===')
for item in results['FILE_ONLY_MATCH'][:15]:
    print(f'  {item["cr"]} | {item["node"]}')
if len(results['FILE_ONLY_MATCH']) > 15:
    print(f'  ... and {len(results["FILE_ONLY_MATCH"]) - 15} more')

print()
print('=== NO_FILE_MATCH (File not indexed) ===')
for item in results['NO_FILE_MATCH']:
    print(f'  {item["cr"]} | {item["node"]}')

conn.close()
