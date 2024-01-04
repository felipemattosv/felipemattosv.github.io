import React from "react"
import useDocusaurusContext from "@docusaurus/useDocusaurusContext"
import { Redirect } from "@docusaurus/router"

export default function Home() {
  const context = useDocusaurusContext()
  const { siteConfig = {} } = context

  return <Redirect to="/intro" />
}
