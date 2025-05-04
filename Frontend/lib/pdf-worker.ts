import { pdfjs } from 'react-pdf';

export const setPdfWorker = () => {
  // In a Next.js environment, we need to handle the PDF worker differently
  // based on whether we're on the client or server side
  if (typeof window === 'undefined') {
    // Skip initialization on the server side
    return;
  }

  // Check if the worker has already been set
  if (pdfjs.GlobalWorkerOptions.workerSrc) {
    return;
  }

  try {
    // Use a reliable CDN for the PDF.js worker
    const workerUrl = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
  } catch (error) {
    console.error('Failed to set PDF worker:', error);
  }
};