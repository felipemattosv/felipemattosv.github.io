/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  aboutSidebar: [{ type: 'autogenerated', dirName: '.' }],

  docsSidebar: [
    {
      type: 'category',
      label: 'Docs',
      items: ['Docs/Docs Overview'],
    },
  ],
  projectsSidebar: [
    {
      type: 'category',
      label: 'Portfolio',
      items: ['Portfolio/Projects Overview', 'Portfolio/ants-bees'],
    },
  ],
};

export default sidebars;
