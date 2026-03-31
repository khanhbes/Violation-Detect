# Firestore React Pagination Kit

Snippet này triển khai đúng plan:
- Pagination `10` docs/lần (`limit(10)` + `startAfter(lastDoc)`).
- Dùng `summary collection` (`items_list`) để chỉ tải field list cần thiết.
- Bật offline persistence 1 lần khi app boot.
- Chặn vòng lặp vô tận trong `useEffect`.

## Files
- `firebase/firestoreClient.ts`: init Firestore + `ensureFirestorePersistence()`.
- `hooks/usePaginatedList.ts`: hook `usePaginatedList(...)`.
- `services/itemsService.ts`: lấy dữ liệu detail từ `items/{id}`.
- `components/ItemsListExample.tsx`: ví dụ UI `Load more`.

## Cách dùng nhanh
1. Copy thư mục này vào project React của bạn (ví dụ `src/firestore`).
2. Cập nhật `firebaseConfig` trong `firebase/firestoreClient.ts` bằng biến môi trường thật.
3. Ở entrypoint app (`App.tsx` hoặc `main.tsx`), gọi 1 lần:

```ts
import { ensureFirestorePersistence } from './firestore/firebase/firestoreClient';

void ensureFirestorePersistence();
```

4. Ở list screen:

```tsx
const { items, loading, error, hasMore, refresh, loadMore } = usePaginatedList({
  path: 'items_list',
  pageSize: 10,
  orderField: 'createdAt',
  orderDir: 'desc',
  requiredFields: ['title', 'status', 'createdAt', 'thumbnail'],
});
```

## Lưu ý quan trọng
- Firestore Web SDK không hỗ trợ query projection kiểu `select(field1, field2)`.
- Để giảm băng thông thật sự, cần collection list nhẹ (`items_list`) như plan.
- Hook đã có:
  - `isFetchingRef` để chặn gọi chồng.
  - `initializedRef` + `queryKey` ổn định để tránh loop `useEffect`.
  - dedupe theo `id` khi append.
