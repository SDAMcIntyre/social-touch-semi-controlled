# Idea: Investigate White Color Points in Forearm Pointcloud Extraction

**Date:** 2026-03-16
**Status:** Idea

## Summary

Forearm pointcloud extraction is producing points with white colors (likely RGB 255,255,255 or similar edge/invalid values). This suggests either a color mapping issue, missing data handling, or incorrect frame masking during pointcloud generation.

## Rough Approach

- Review the pointcloud generation code in `code/src/analysis/` (likely `kinect_processor` or similar)
- Check how colors are being assigned to points (RGB mapping from frame data)
- Verify masking/filtering logic for valid vs. invalid points
- Test with sample forearm data to reproduce the white points

## Notes

- Check if this is a rendering artifact (white = invalid/masked in visualization) or actual data issue
- Related: may intersect with touch analytics matrix generation pipeline
- Open question: does this occur in specific conditions (certain sessions, specific body regions)?
