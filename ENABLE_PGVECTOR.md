# Enable pgvector Extension in Supabase

The pgvector extension is required for vector similarity search in RAG-Anything. Here are multiple ways to enable it:

## Method 1: Automated Script (Recommended)

Run the automated setup script:

```bash
python setup_pgvector.py
```

This will:
- Connect to your Supabase database
- Check if pgvector exists
- Create it if it doesn't exist
- Verify the installation

## Method 2: Supabase SQL Editor (Manual)

If the automated script doesn't work (due to permissions), use the Supabase SQL Editor:

1. **Go to your Supabase Dashboard**:
   - https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk

2. **Navigate to SQL Editor**:
   - Click on "SQL Editor" in the left sidebar
   - Or go directly to: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new

3. **Run this SQL command**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Click "Run"** to execute the command

5. **Verify it worked**:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```
   
   You should see a row with the vector extension.

## Method 3: Using psql Command Line

If you have `psql` installed and database credentials:

```bash
psql "postgresql://postgres:YOUR_PASSWORD@db.ngbhnjvojqqesacnijwk.supabase.co:5432/postgres" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Verification

After enabling pgvector, verify it's working:

```python
from raganything import create_pgvector_extension, get_supabase_url_from_jwt

SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY)

# This should return True if extension exists
result = create_pgvector_extension(
    supabase_url=SUPABASE_URL,
    supabase_db_password="your_password"
)
print("Extension ready!" if result else "Extension not found")
```

## Troubleshooting

### "Insufficient Privileges" Error

If you get a privileges error, you need to use the Supabase SQL Editor (Method 2) instead of direct database connection.

### "Extension Already Exists"

If you see this message, pgvector is already enabled! You're good to go.

### Connection Issues

- Verify your database password is correct
- Check Supabase Dashboard → Settings → Database for connection details
- Ensure your IP is not blocked

## Quick Links

- **Your Project Dashboard**: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk
- **SQL Editor**: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new
- **Database Settings**: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/settings/database

## Next Steps

Once pgvector is enabled:

1. ✅ Run `python setup_pgvector.py` to verify
2. ✅ Use `python examples/supabase_quickstart.py` to test RAG-Anything
3. ✅ Start processing documents and storing vectors in Supabase!


