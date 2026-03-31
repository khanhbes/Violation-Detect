import { FirebaseError, getApp, getApps, initializeApp } from 'firebase/app';
import {
  Firestore,
  enableIndexedDbPersistence,
  enableMultiTabIndexedDbPersistence,
  getFirestore,
} from 'firebase/firestore';

/**
 * Replace values with your own environment-based config in real app code.
 * This sample keeps placeholders so it is safe to commit.
 */
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY ?? '',
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN ?? '',
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID ?? '',
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET ?? '',
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID ?? '',
  appId: process.env.REACT_APP_FIREBASE_APP_ID ?? '',
};

const firebaseApp = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
export const db = getFirestore(firebaseApp);

let persistenceInitPromise: Promise<void> | null = null;
let persistenceReady = false;

function isFirebaseCode(error: unknown, code: string): boolean {
  return error instanceof FirebaseError && error.code === code;
}

/**
 * Call once at app bootstrap.
 * - Try multi-tab persistence first.
 * - Fallback to single-tab persistence.
 * - If unavailable/already enabled, continue online without throwing.
 */
export function ensureFirestorePersistence(firestore: Firestore = db): Promise<void> {
  if (persistenceReady) return Promise.resolve();
  if (persistenceInitPromise) return persistenceInitPromise;

  persistenceInitPromise = (async () => {
    try {
      await enableMultiTabIndexedDbPersistence(firestore);
      persistenceReady = true;
      return;
    } catch (error) {
      if (!isFirebaseCode(error, 'failed-precondition')) {
        // Continue trying single-tab persistence for unsupported/other cases.
      }
    }

    try {
      await enableIndexedDbPersistence(firestore);
      persistenceReady = true;
    } catch (error) {
      // Do not crash app if persistence is not available in this environment.
      if (
        isFirebaseCode(error, 'failed-precondition') ||
        isFirebaseCode(error, 'unimplemented')
      ) {
        return;
      }
      throw error;
    }
  })();

  return persistenceInitPromise;
}
