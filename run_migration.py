"""
Run a SQL migration file
"""
import sys
from config.settings import get_supabase_client

def run_migration(sql_file_path: str):
    """Execute SQL migration"""
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    supabase = get_supabase_client()
    
    # Split by statements and execute each
    statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
    
    for i, stmt in enumerate(statements):
        if not stmt:
            continue
        print(f"Executing statement {i+1}/{len(statements)}...")
        try:
            result = supabase.rpc('exec_sql', {'sql': stmt}).execute()
            print(f"✓ Statement {i+1} executed successfully")
        except Exception as e:
            print(f"✗ Statement {i+1} failed: {e}")
            print(f"Statement: {stmt[:200]}...")
            # Continue with other statements
    
    print(f"\nMigration completed: {sql_file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_migration.py <migration_file.sql>")
        sys.exit(1)
    
    run_migration(sys.argv[1])

