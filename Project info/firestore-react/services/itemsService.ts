import { doc, getDoc } from 'firebase/firestore';
import { db } from '../firebase/firestoreClient';

/**
 * Convention from the plan:
 * - List screen reads from `items_list`
 * - Detail screen reads from `items/{id}`
 */
export async function fetchItemDetail(itemId: string): Promise<Record<string, unknown> | null> {
  const normalizedId = itemId.trim();
  if (!normalizedId) return null;

  const ref = doc(db, 'items', normalizedId);
  const snap = await getDoc(ref);
  if (!snap.exists()) return null;
  return { id: snap.id, ...snap.data() };
}
