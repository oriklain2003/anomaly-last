import html2canvas from 'html2canvas';

/**
 * Capture a screenshot of the full application window.
 * This captures everything visible in the viewport.
 */
export async function captureFullWindow(): Promise<string> {
    // Find the main app container or use document body
    const appRoot = document.getElementById('root') || document.body;
    
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    
    const canvas = await html2canvas(appRoot, {
        scale: scale,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#0f0f1a', // Match app dark background
        logging: false,
        width: window.innerWidth,
        height: window.innerHeight,
        windowWidth: window.innerWidth,
        windowHeight: window.innerHeight,
        x: 0,
        y: 0,
        scrollX: 0,
        scrollY: 0,
        // Handle WebGL canvases (MapLibre)
        onclone: (clonedDoc) => {
            // Try to copy the WebGL canvas content to the cloned document
            const originalMapCanvas = document.querySelector('.maplibregl-canvas') as HTMLCanvasElement;
            const clonedMapCanvas = clonedDoc.querySelector('.maplibregl-canvas') as HTMLCanvasElement;
            
            if (originalMapCanvas && clonedMapCanvas) {
                try {
                    const ctx = clonedMapCanvas.getContext('2d');
                    if (ctx) {
                        // Copy the WebGL canvas content
                        ctx.drawImage(originalMapCanvas, 0, 0);
                    }
                } catch (e) {
                    console.warn('Could not copy WebGL canvas:', e);
                }
            }
        }
    });

    return canvas.toDataURL('image/png');
}

/**
 * Capture a screenshot using the browser's Screen Capture API.
 * This requires user permission but captures the actual screen content.
 */
export async function captureScreenWithPermission(): Promise<string | null> {
    try {
        // Request screen capture permission
        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: {
                displaySurface: 'browser'
            } as MediaTrackConstraints,
            audio: false
        });
        
        // Get the video track
        const track = stream.getVideoTracks()[0];
        
        // Create a video element to capture from
        const video = document.createElement('video');
        video.srcObject = stream;
        await video.play();
        
        // Wait for video to be ready
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Create canvas and draw the video frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
            track.stop();
            return null;
        }
        
        ctx.drawImage(video, 0, 0);
        
        // Stop the stream
        track.stop();
        
        return canvas.toDataURL('image/png');
    } catch (e) {
        console.error('Screen capture failed:', e);
        return null;
    }
}

/**
 * Capture a screenshot of a DOM element and return it as a base64 PNG string.
 * Handles high-DPI displays for quality screenshots.
 */
export async function captureScreenshot(element: HTMLElement): Promise<string> {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    
    const canvas = await html2canvas(element, {
        scale: scale,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#1a1a2e',
        logging: false
    });

    return canvas.toDataURL('image/png');
}

/**
 * Capture the map with overlays and controls.
 * Combines the WebGL canvas with DOM overlays.
 */
export async function captureMapWithOverlays(mapContainer: HTMLElement): Promise<string> {
    const rect = mapContainer.getBoundingClientRect();
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    
    const combinedCanvas = document.createElement('canvas');
    combinedCanvas.width = rect.width * scale;
    combinedCanvas.height = rect.height * scale;
    const ctx = combinedCanvas.getContext('2d');
    
    if (!ctx) {
        throw new Error('Failed to get canvas context');
    }
    
    ctx.scale(scale, scale);
    
    // Try to capture the WebGL canvas first
    const mapCanvas = mapContainer.querySelector('.maplibregl-canvas') as HTMLCanvasElement;
    
    if (mapCanvas) {
        try {
            ctx.drawImage(mapCanvas, 0, 0, rect.width, rect.height);
        } catch (e) {
            console.warn('Failed to draw map canvas:', e);
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, rect.width, rect.height);
        }
    }
    
    // Now capture the DOM overlays using html2canvas
    try {
        const overlayCanvas = await html2canvas(mapContainer, {
            scale: scale,
            useCORS: true,
            backgroundColor: 'transparent',
            logging: false,
            ignoreElements: (element) => {
                return element.classList?.contains('maplibregl-canvas');
            }
        });
        
        ctx.drawImage(overlayCanvas, 0, 0, rect.width, rect.height);
    } catch (e) {
        console.warn('Failed to capture overlays:', e);
    }
    
    return combinedCanvas.toDataURL('image/png');
}

/**
 * Get base64 data without the data URL prefix.
 */
export function stripDataUrlPrefix(dataUrl: string): string {
    const prefix = 'data:image/png;base64,';
    if (dataUrl.startsWith(prefix)) {
        return dataUrl.slice(prefix.length);
    }
    return dataUrl;
}
