import { useEffect } from 'react';
import { ensureFirestorePersistence } from '../firebase/firestoreClient';
import { usePaginatedList } from '../hooks/usePaginatedList';

export function ItemsListExample() {
  useEffect(() => {
    void ensureFirestorePersistence();
  }, []);

  const { items, loading, error, hasMore, refresh, loadMore } = usePaginatedList({
    path: 'items_list',
    pageSize: 10,
    orderField: 'createdAt',
    orderDir: 'desc',
    requiredFields: ['title', 'status', 'createdAt', 'thumbnail'],
    enabled: true,
  });

  return (
    <section>
      <header style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <button onClick={() => void refresh()} disabled={loading}>
          Refresh
        </button>
      </header>

      {error && <p style={{ color: 'crimson' }}>Error: {error}</p>}

      <ul style={{ listStyle: 'none', padding: 0 }}>
        {items.map((item) => (
          <li key={item.id} style={{ marginBottom: 10 }}>
            <strong>{String(item.title ?? '(no title)')}</strong>
            <div>Status: {String(item.status ?? '-')}</div>
            <div>Created: {String(item.createdAt ?? '-')}</div>
          </li>
        ))}
      </ul>

      <button onClick={() => void loadMore()} disabled={loading || !hasMore}>
        {loading ? 'Loading...' : hasMore ? 'Load more' : 'No more items'}
      </button>
    </section>
  );
}
