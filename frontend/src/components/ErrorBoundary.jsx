import { Component } from 'react'

/* Without this, an uncaught error anywhere in the tree unmounts the whole app - every
 * page goes blank at once, silently, with nothing on screen to say why. This turns that
 * into a visible message plus a reload button, and logs the real error to the console so
 * it can be diagnosed instead of just reported as "the site went blank". */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('Unhandled error - the app fell back to this boundary:', error, info)
  }

  render() {
    if (!this.state.error) return this.props.children

    return (
      <div className="flex min-h-screen items-center justify-center bg-ink-950 p-4">
        <div className="w-full max-w-md rounded-2xl border border-ink-700 bg-ink-900 p-6 text-center">
          <p className="text-[15px] font-semibold text-ink-50">Something went wrong.</p>
          <p className="mt-2 text-[12.5px] text-ink-400">
            {this.state.error.message || 'An unexpected error occurred.'}
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-4 rounded-lg bg-brand-500 px-4 py-2 text-[13px] font-semibold text-on-brand transition-opacity hover:opacity-85"
          >
            Reload
          </button>
        </div>
      </div>
    )
  }
}
