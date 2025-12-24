"""
Apply SQL migration using Supabase SQL editor or direct execution
Since Supabase doesn't support arbitrary SQL via client, this script helps format the SQL
"""
import sys

def format_migration(sql_file_path: str):
    """Read and display SQL for manual execution in Supabase SQL editor"""
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    print("=" * 80)
    print(f"Migration file: {sql_file_path}")
    print("=" * 80)
    print("\nCopy and paste the SQL below into your Supabase SQL Editor:")
    print("=" * 80)
    print(sql)
    print("=" * 80)
    print("\nTo apply:")
    print("1. Go to your Supabase Dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Paste the SQL above")
    print("4. Click 'Run'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python apply_migration.py <migration_file.sql>")
        print("\nAvailable migrations:")
        import os
        migrations_dir = "migrations"
        if os.path.exists(migrations_dir):
            for f in sorted(os.listdir(migrations_dir)):
                if f.endswith('.sql'):
                    print(f"  - {os.path.join(migrations_dir, f)}")
        sys.exit(1)
    
    format_migration(sys.argv[1])

