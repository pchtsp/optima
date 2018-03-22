library(shiny)
library(shinydashboard)
library(timevis)
library(DT)
library(plotly)

dashboardPage(
    dashboardHeader(title = "OptiPlan Demo"),
    dashboardSidebar(
        sidebarMenu(
            menuItem("Données d'entreé", tabName = "input_data", icon = icon("dashboard")),
            menuItem("Résultats", tabName = "results", icon = icon("th"))
        )
    ),
    dashboardBody(
        tabItems(
            tabItem(tabName = "input_data",
                    selectInput("selectExp", label = h3("Etude"), 
                                choices = list(),
                                selectize = T
                                ),
                    fluidRow(
                        column(width = 4, 
                               h3("Parmètres"),
                               DT::dataTableOutput("parameters")),
                        column(width = 6, 
                               h3("Missions"),
                               DT::dataTableOutput("missions"), offset = 2)
                    )
            ),
            tabItem(tabName = "results",
                    # h2("Planning"),
                    timevisOutput('timevis1', height = 500, width="95%"),
                    # radioButtons(inputId='option_graph',
                    #              label='Type de graph',
                    #              choices=c('Cravatte'='tie', 'Evolution'='evolution')),
                    h4(textOutput('resource')),
                    plotlyOutput('plotly1')
                    
            ),
            tabItem(tabName = "generation",
                    h2("Résolution"),
                    dateRangeInput('StartEnd', label = 'Start end', format = "yyy-mm", language='fr'),
                    actionButton(inputId='solve', label='Résoudre'),
                    textAreaInput('solve_log', 'Log')
            )
        )
    )
)
