# hexo-sources-env
This branch stores my customized yilia themes and blog origin markdown files for migration.

## Usage
- download all the node_modules
```bash
npm install
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
