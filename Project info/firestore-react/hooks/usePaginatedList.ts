import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  collection,
  getDocs,
  limit,
  orderBy,
  query,
  QueryConstraint,
  QueryDocumentSnapshot,
  startAfter,
  where,
  WhereFilterOp,
} from 'firebase/firestore';
import { db } from '../firebase/firestoreClient';

export type ListFilter = {
  field: string;
  op: WhereFilterOp;
  value: unknown;
};

export type UsePaginatedListOptions = {
  path: string;
  pageSize?: number;
  orderField?: string;
  orderDir?: 'asc' | 'desc';
  filters?: ListFilter[];
  requiredFields?: string[];
  enabled?: boolean;
};

export type ListItem = {
  id: string;
  [key: string]: unknown;
};

export type UsePaginatedListResult = {
  items: ListItem[];
  loading: boolean;
  error: string | null;
  hasMore: boolean;
  refresh: () => Promise<void>;
  loadMore: () => Promise<void>;
};

function pickFields(
  doc: QueryDocumentSnapshot,
  requiredFields: string[] | undefined,
): ListItem {
  const data = doc.data();
  if (!requiredFields || requiredFields.length === 0) {
    return { id: doc.id, ...data };
  }

  const projected: ListItem = { id: doc.id };
  for (const field of requiredFields) {
    if (Object.prototype.hasOwnProperty.call(data, field)) {
      projected[field] = data[field];
    }
  }
  return projected;
}

function dedupeById(items: ListItem[]): ListItem[] {
  const map = new Map<string, ListItem>();
  for (const item of items) {
    map.set(item.id, item);
  }
  return Array.from(map.values());
}

/**
 * Firestore paginated list hook:
 * - load first page with limit(10)
 * - load more via startAfter(lastVisibleDoc)
 * - prevent duplicated fetches + useEffect loops
 */
export function usePaginatedList({
  path,
  pageSize = 10,
  orderField = 'createdAt',
  orderDir = 'desc',
  filters = [],
  requiredFields = [],
  enabled = true,
}: UsePaginatedListOptions): UsePaginatedListResult {
  const [items, setItems] = useState<ListItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState<boolean>(true);

  const mountedRef = useRef<boolean>(false);
  const isFetchingRef = useRef<boolean>(false);
  const initializedRef = useRef<boolean>(false);
  const lastDocRef = useRef<QueryDocumentSnapshot | null>(null);
  const hasMoreRef = useRef<boolean>(true);
  const requestSeqRef = useRef<number>(0);
  const currentQueryKeyRef = useRef<string>('');

  const filtersKey = JSON.stringify(
    filters.map((f) => ({ field: f.field, op: f.op, value: f.value })),
  );
  const fieldsKey = JSON.stringify(requiredFields);
  const queryKey = `${path}|${pageSize}|${orderField}|${orderDir}|${filtersKey}|${fieldsKey}|${enabled}`;

  const stableFilters = useMemo<ListFilter[]>(
    () => filters.map((f) => ({ field: f.field, op: f.op, value: f.value })),
    [filtersKey],
  );

  const resetCursor = useCallback(() => {
    lastDocRef.current = null;
    hasMoreRef.current = true;
    setHasMore(true);
  }, []);

  const fetchPage = useCallback(
    async (mode: 'replace' | 'append') => {
      if (!enabled) return;
      if (isFetchingRef.current) return;
      if (mode === 'append' && !hasMoreRef.current) return;

      isFetchingRef.current = true;
      setLoading(true);
      setError(null);
      const requestId = ++requestSeqRef.current;

      try {
        const constraints: QueryConstraint[] = [];
        for (const f of stableFilters) {
          constraints.push(where(f.field, f.op, f.value));
        }
        constraints.push(orderBy(orderField, orderDir));
        constraints.push(limit(pageSize));

        if (mode === 'append' && lastDocRef.current) {
          constraints.push(startAfter(lastDocRef.current));
        }

        const q = query(collection(db, path), ...constraints);
        const snapshot = await getDocs(q);

        if (!mountedRef.current || requestId !== requestSeqRef.current) {
          return;
        }

        const incoming = snapshot.docs.map((doc) => pickFields(doc, requiredFields));
        lastDocRef.current =
          snapshot.docs.length > 0
            ? snapshot.docs[snapshot.docs.length - 1]
            : mode === 'replace'
              ? null
              : lastDocRef.current;

        const nextHasMore = snapshot.docs.length === pageSize;
        hasMoreRef.current = nextHasMore;
        setHasMore(nextHasMore);

        if (mode === 'replace') {
          setItems(dedupeById(incoming));
        } else {
          setItems((prev) => dedupeById([...prev, ...incoming]));
        }
      } catch (err) {
        if (!mountedRef.current || requestId !== requestSeqRef.current) return;
        const message = err instanceof Error ? err.message : 'Unknown Firestore error';
        setError(message);
      } finally {
        if (mountedRef.current && requestId === requestSeqRef.current) {
          setLoading(false);
        }
        isFetchingRef.current = false;
      }
    },
    [enabled, orderDir, orderField, pageSize, path, requiredFields, stableFilters],
  );

  const refresh = useCallback(async () => {
    resetCursor();
    await fetchPage('replace');
  }, [fetchPage, resetCursor]);

  const loadMore = useCallback(async () => {
    await fetchPage('append');
  }, [fetchPage]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Prevent infinite loop:
  // - Depend only on stable queryKey + fetch callback
  // - Do not depend on items/loading states here
  useEffect(() => {
    if (!enabled) return;

    const queryChanged = currentQueryKeyRef.current !== queryKey;
    if (!queryChanged && initializedRef.current) return;

    currentQueryKeyRef.current = queryKey;
    initializedRef.current = true;
    resetCursor();
    setItems([]);
    void fetchPage('replace');
  }, [enabled, fetchPage, queryKey, resetCursor]);

  return { items, loading, error, hasMore, refresh, loadMore };
}
