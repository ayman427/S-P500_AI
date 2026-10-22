import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { ClerkProvider } from '@clerk/react'
import AuthGate from './AuthGate'

const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {publishableKey ? <ClerkProvider publishableKey={publishableKey}>
      <AuthGate />
    </ClerkProvider> : <main className="auth-page"><p role="alert">Sign-in is not configured yet.</p></main>}
  </StrictMode>,
)
