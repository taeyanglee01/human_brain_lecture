"""
List files in OpenNeuro dataset ds000221 to understand structure
"""
import requests
import json

# OpenNeuro GraphQL API
API_URL = "https://openneuro.org/crn/graphql"

# Query to get file tree
query = """
query {
  dataset(id: "ds000221") {
    id
    snapshots {
      tag
      files {
        id
        filename
        size
      }
    }
  }
}
"""

print("Querying OpenNeuro API for ds000221 file structure...")
print("="*60)

try:
    response = requests.post(API_URL, json={'query': query})
    data = response.json()

    if 'errors' in data:
        print("API Error:", data['errors'])
    else:
        dataset = data['data']['dataset']
        print(f"Dataset ID: {dataset['id']}")
        print(f"\nSnapshots available: {len(dataset['snapshots'])}")

        # Get the latest snapshot
        if dataset['snapshots']:
            snapshot = dataset['snapshots'][0]
            print(f"Using snapshot: {snapshot['tag']}")
            print(f"\nTotal files: {len(snapshot['files'])}")

            # Filter files for sub-01 and sub-15
            print("\n" + "="*60)
            print("Files for sub-01:")
            print("="*60)
            sub01_files = [f for f in snapshot['files'] if 'sub-01' in f['filename']]
            for f in sub01_files[:20]:  # Show first 20
                print(f"  {f['filename']} ({f['size']} bytes)")
            if len(sub01_files) > 20:
                print(f"  ... and {len(sub01_files) - 20} more files")

            print("\n" + "="*60)
            print("Files for sub-15:")
            print("="*60)
            sub15_files = [f for f in snapshot['files'] if 'sub-15' in f['filename']]
            for f in sub15_files[:20]:
                print(f"  {f['filename']} ({f['size']} bytes)")
            if len(sub15_files) > 20:
                print(f"  ... and {len(sub15_files) - 20} more files")

            # Save full file list to JSON
            output_file = r'C:\Users\Public\human_brain_lecture\data\ds000221_files.json'
            with open(output_file, 'w') as f:
                json.dump(snapshot['files'], f, indent=2)
            print(f"\n\nFull file list saved to: {output_file}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
