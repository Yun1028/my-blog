import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const projects = defineCollection({
  loader: glob({
    pattern: '**/index.{md,mdx}',
    base: './src/content/projects',
  }),
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    description: z.string(),
    summary: z.string().optional(),
    status: z.enum(['ongoing', 'completed']).default('ongoing'),
    coverImage: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

const projectUpdates = defineCollection({
  loader: glob({
    pattern: '**/updates/*.{md,mdx}',
    base: './src/content/projects',
  }),
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    description: z.string(),
    youtube: z.string().url().optional(),
  }),
});

const research = defineCollection({
  loader: glob({
    pattern: '**/*.{md,mdx}',
    base: './src/content/research',
  }),
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    description: z.string(),
    summary: z.string().optional(),
    status: z.enum(['ongoing', 'paused', 'finished']).default('ongoing'),
    tags: z.array(z.string()).default([]),

    overview: z.string(),
    hypothesis: z.string(),
    method: z.array(z.string()).default([]),
    experimentDesign: z.array(z.string()).default([]),

    timeline: z.array(
      z.object({
        date: z.coerce.date(),
        title: z.string(),
        status: z.enum(['ongoing', 'paused', 'finished']).optional(),
        summary: z.string(),
        details: z.array(z.string()).default([]),
      })
    ).default([]),
  }),
});

const study = defineCollection({
  loader: glob({
    pattern: '**/*.{md,mdx}',
    base: './src/content/study',
  }),
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    description: z.string(),
    ppt: z.string().url().optional(),
  }),
});

export const collections = {
  projects,
  projectUpdates,
  research,
  study,
};