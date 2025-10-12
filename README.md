# hexo-sources-env
This branch stores my customized yilia themes and blog origin markdown files for migration.
**Note:** `source/_drafts/` and `source/_discarded/` are hidden and ignored.

## Install
- download `nvm` and install npm.
- install `npm install -g pnpm`.
- download hexo-cli `pnpm add -g hexo-cli`.
- download rimraf tools `pnpm add -g rimraf`.
- add `.npmrc` for registry

## Usage
- download all the node_modules
```bash
pnpm install
```

- git add new files to this branch
```bash
git add .
git commit -m "update"
git push origin hexo-sources-env
```

- deploy command
```bash
hexo clean
hexo g
hexo s # you can check and deploy in localhost
hexo d # push to remote repository for deployment
```

- uninstall node_modules?
```bash
rimraf node_modules
```