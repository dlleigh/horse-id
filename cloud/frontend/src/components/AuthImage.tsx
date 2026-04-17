import { useEffect, useState } from 'react'
import { getJWTToken } from '../lib/auth'

/** <img> wrapper that appends the JWT token as a query param so auth works. */
export default function AuthImage({ src, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) {
  const [authSrc, setAuthSrc] = useState<string | undefined>()

  useEffect(() => {
    let cancelled = false
    if (!src) return
    getJWTToken().then(token => {
      if (cancelled) return
      if (token) {
        const sep = src.includes('?') ? '&' : '?'
        setAuthSrc(`${src}${sep}token=${encodeURIComponent(token)}`)
      } else {
        setAuthSrc(src)
      }
    })
    return () => { cancelled = true }
  }, [src])

  if (!authSrc) return null
  return <img src={authSrc} {...props} />
}
