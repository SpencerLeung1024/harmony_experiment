"""
Harmony From First Principles - Gradio Web Interface

Main entry point for launching the web interface.
Run with: python app.py

For HuggingFace Spaces deployment, this file will be used as the entry point.
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harmony.ui import GradioInterface


def main():
    """Launch the Gradio web interface."""
    parser = argparse.ArgumentParser(
        description="Harmony From First Principles - Web Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port to bind to"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link (for HuggingFace Spaces)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Authentication credentials as 'username:password'"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    print("=" * 60)
    print("🎵 Harmony From First Principles")
    print("=" * 60)
    print()
    print("Initializing Gradio interface...")
    
    # Create interface
    demo = GradioInterface()
    interface = demo.create_interface()
    
    print("Interface created successfully!")
    print()
    
    # Authentication
    auth = None
    if args.auth:
        username, password = args.auth.split(":")
        auth = (username, password)
    
    # Launch
    print(f"Launching server on {args.host}:{args.port}...")
    print(f"Share: {args.share}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        auth=auth,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
